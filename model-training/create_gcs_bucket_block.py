import os
# from prefect_gcp import GcsBucket
from prefect.filesystems import GCS

block = GCS(bucket_path="my-bucket/folder/")

PREFECT_FLOWS_BUCKET = "citibike-training-prefect-flows"

def create_gcs_bucket_block(bucket_name: str, folder: str):
    gcs_bucket_block = GCS(bucket_path=f"{bucket_name}/{folder}")
    # gcs_bucket_block = GcsBucket(
    #     bucket=bucket_name
    # )
    gcs_bucket_block.save(name=bucket_name, overwrite=True)

if __name__ == "__main__":
    create_gcs_bucket_block(bucket_name=PREFECT_FLOWS_BUCKET, folder="model-training") # block_name = "citibike-training-bucket"