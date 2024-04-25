# Databricks notebook source
# MAGIC %pip install transformers>=4.39.2

# COMMAND ----------

# MAGIC %pip install ray[default]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip list

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES

GPUS_PER_NODE = 4
NUM_OF_WORKER_NODES = 8

setup_ray_cluster(
  num_cpus_worker_node=1,
  num_gpus_per_node = GPUS_PER_NODE,
  max_worker_nodes = NUM_OF_WORKER_NODES,
  num_cpus_head_node=1,
  collect_log_to_path="/dbfs/tmp/raylogs",
)

# COMMAND ----------

import ray
ray.init()
ray.cluster_resources()

# COMMAND ----------


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.cuda.amp import autocast
import numpy as np
from typing import Dict, List
import re
from transformers import file_utils

class TextSummarizer:

    def __init__(self,checkpoint="meta-llama/Meta-Llama-3-8B-Instruct", verbose=False):
      # Initialize the tokenizer and model on each worker
      print("Initialize the tokenizer and model on each worker")
      self.checkpoint = checkpoint
      self.access_token = 'your hf token retrieved from secret scope'
      self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True, token=self.access_token)
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model = None
      gpu_ids = ray.get_gpu_ids()
      print(f"allocated gpu ids: {gpu_ids}")
      self.verbose = verbose

    def _create_model(self):
      if self.model: return
      self.model = AutoModelForCausalLM.from_pretrained(
          self.checkpoint,
          torch_dtype=torch.float16, 
          trust_remote_code=True,
          device_map="auto",
          token=self.access_token
      )
      self.model.eval()

    def chunk(self, text: str, chunk_size: int = 1024, overlap: int = 0) -> List[str]:
      """
      Chunk a large text by sentences, excluding empty sentences from the chunking calculation, ensuring each chunk has a
      number of tokens close to or less than the specified chunk size, with an option to have overlapping sentences
      between chunks.
      
      Parameters:
      - text: The input text to be chunked.
      - chunk_size: The desired maximum number of tokens per chunk (default 1024).
      - overlap: The number of sentences to overlap between consecutive chunks.
      
      Returns:
      - A list of text chunks, each consisting of full sentences and with a token count close to the specified chunk size.
      """
      
      # Split the text into sentences and filter out any empty sentences
      sentences = [sentence for sentence in re.split(r'(?<=[.!?]) +', text) if sentence.strip()]
      
      # Tokenize the text to count tokens
      tokens = text.split()
      total_tokens = len(tokens)
      
      # Calculate how many sentences approximately go into each chunk to respect the token limit
      if total_tokens > chunk_size:
          sentences_per_chunk = max(1, len(sentences) // (total_tokens // chunk_size))
      else:
          sentences_per_chunk = len(sentences)
      
      # Adjust sentences per chunk based on overlap, ensuring we don't exceed the total number of sentences
      effective_sentences_per_chunk = max(1, sentences_per_chunk - overlap)
      
      # Chunk the text by sentences based on the calculated sentences per chunk, including overlap
      chunks = []
      for i in range(0, len(sentences), effective_sentences_per_chunk):
          chunk_end = min(i + sentences_per_chunk, len(sentences))
          chunk = " ".join(sentences[i:chunk_end])
          chunks.append(chunk)
          
          # Prevent creating a chunk that is solely overlap with no new sentences
          if chunk_end == len(sentences):
              break
          
      return chunks

    def chunk_text_by_tokens(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """
        Chunk a text into segments based on a specific token count, with an option for overlapping tokens between chunks.
        
        Parameters:
        - text: The input text to be chunked.
        - chunk_size: The desired number of tokens per chunk.
        - overlap: The number of tokens to overlap between consecutive chunks.
        
        Returns:
        - A list of text chunks, each with a specified number of tokens, potentially overlapping.
        """
        
        # Tokenize the text. Here, a simple split is used, assuming whitespace tokenization.
        tokens = text.split()
        
        # Ensure chunk_size is at least 1 to avoid infinite loop
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
            
        # Ensure overlap is not larger than chunk_size to avoid infinite looping or empty chunks
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        
        chunks = []
        i = 0
        
        while i < len(tokens):
            # Determine the end of the current chunk, not exceeding the total token count
            end = min(i + chunk_size, len(tokens))
            
            # Extract the chunk based on current indices
            chunk_tokens = tokens[i:end]
            chunks.append(" ".join(chunk_tokens))
            
            # Move the start of the next chunk, allowing for overlap
            i += chunk_size - overlap
        
        return chunks

    def _extract_output(self, response: str):
      result = ""
      output_parts = response.split("\nOutput:", 1)
      if len(output_parts) >= 2:
        followup_parts = output_parts[1].split("\n\nFollow up")
        if len(followup_parts) >= 1:
          result = followup_parts[0].strip()
        else:
          result = output_parts[1].strip()
      return result if result else response

    # Do not call this method directly. The txt needs to contain the summarizing prompt.
    def _summarize(self, txt: str):
      try:
        input_text = txt 
        with torch.no_grad():
          input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
          outputs = self.model.generate(input_ids, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_output(summary)
      except RuntimeError as e:
        return ""
    
    def map_summary(self, txt: str):
      if self.verbose:
        print(f"Map summary: {txt}")
      input_text = f"Instruct: Summarize the following medical report. Please stick to the original content of the report and do not add any superfluous information.\nReport: {txt}\nOutput:"
      return self._summarize(input_text)
    
    def reduce_summary(self, summaries: List[str]):
      txt = "\n\n".join(summaries)
      if self.verbose:
        print(f"Reduce summaries: {txt}")
      input_text = f"Instruct: Summarize the following medical report. Please stick to the original content of the report and do not add any superfluous information.\nReport: {txt}\nOutput:"
      return self._summarize(input_text)
    
    def refine_summary(self, existingSummary: str, txt: str):
      if self.verbose:
        print(f"Refine summary: {existingSummary}. More text: {txt}")
      input_txt = f"Instruct: Produce a final summary from an existing summary and new context into a final summary of the medical report.\nExisting summary: {txt}\nNew context: {txt}\nOutput:"
      return self._summarize(input_txt)

    def process_note(self, note):
      chunks = self.chunk_text_by_tokens(note, chunk_size=256, overlap=64)
      summaries = []
      for chunk in chunks:
        summary = self.map_summary(chunk)
        summaries.append(summary)
      print(f"summaries: {len(summaries)}")
      if len(summaries) >= 2:
        # txt = "\n\n".join(summaries)
        final_sum = self.reduce_summary(summaries)
        return final_sum
      else:
        return summaries[0]
      
    def process_note_refine(self, note):
      chunks = self.chunk_text_by_tokens(note, chunk_size=256, overlap=64)
      print(f"chunks {len(chunks)}")
      summaries = [self.map_summary(chunks[0])]
      for i, chunk in enumerate(chunks[1:]):
        summary = self.refine_summary(summaries[i-1], chunk)
        summaries.append(summary)
      return summaries[-1]
    
    def process_note_no_chunking(self, note):
      return self.map_summary(note)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
      import time
      self._create_model()
      summeries = []
      durs = []
      for note in list(batch["reporttext"]):
        #   print(note)
        start_time = time.time()    
        # pred = self.process_note(note)
        # pred = self.process_note_refine(note)
        pred = self.process_note_no_chunking(note)
        if self.verbose: print(f"### Final summary: {pred}")
        summeries.append(pred)
        end_time = time.time()
        dur = end_time - start_time
        durs.append(dur)
      batch["summarized_text"] = summeries
      batch["dur"] = durs
      return batch

# COMMAND ----------

# MAGIC %md ### Read from parquet

# COMMAND ----------

import ray 
rows = ray.data.read_parquet("/dbfs/tmp/data/your parquet file directory").take(16)
ds = ray.data.from_items(rows)

# COMMAND ----------


ray_res = ray.cluster_resources()
num_gpus_per_actor = 4
worker_num = int(ray_res['GPU']/num_gpus_per_actor)
print(f"### The number of workers: {worker_num}")

summarized_ds = ds.map_batches(
  TextSummarizer,
  concurrency=worker_num,
  num_gpus=num_gpus_per_actor,
  batch_size=(ds.count()//worker_num)
)

# COMMAND ----------

summarized_pdf = summarized_ds.to_pandas()

# COMMAND ----------

display(summarized_pdf)

# COMMAND ----------

#summarized_df = spark.createDataFrame(summarized_pdf)
#summarized_df.write.saveAsTable("your table")

# COMMAND ----------

from ray.util.spark import shutdown_ray_cluster
shutdown_ray_cluster()

# COMMAND ----------


