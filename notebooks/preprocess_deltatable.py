# Databricks notebook source
# MAGIC %pip install ray[default]

# COMMAND ----------

import ray
import os
from urllib.parse import urlparse


def create_parquet(spark_dataframe, dbfs_tmp_path):
    spark_dataframe.write.mode('overwrite').parquet(dbfs_tmp_path)
    fuse_path = "/dbfs" + urlparse(dbfs_tmp_path).path
    return fuse_path

# For example, read a Delta Table as a Spark DataFrame
spark_df = spark.read.table("your table name").select("reporttext").where("len(reporttext)>512")

# Provide a dbfs location to write the table to
data_location_2 = (
    "dbfs:/tmp/data/your parquet file directory"
)

# Convert the Spark DataFrame to a Ray dataset
parquet_path = create_parquet(
    spark_dataframe=spark_df,
    dbfs_tmp_path=data_location_2
)

# COMMAND ----------

print(parquet_path)
