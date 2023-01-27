import logging
import os
from pathlib import Path

import boto3
import botocore
import tqdm

# Download a file from s3 bucket to the `data` directory without using credentials
def download_file_from_s3(bucket_name:str, s3_file_path:str, data_dir="data", ):
    s3 = boto3.client('s3', region_name='us-west-2',
                      # config=boto3.session.Config(signature_version='s3v4')
                      )
    s3.download_file(bucket_name, s3_file_path, data_dir)

def get_file_folders(s3_client, bucket_name, prefix=""):
    file_names = []
    folders = []

    default_kwargs = {
        "Bucket": bucket_name,
        "Prefix": prefix
    }
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token

        response = s3_client.list_objects_v2(**updated_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders


def download_files(s3_client, bucket_name, local_path, file_names, folders):
    local_path = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(local_path, folder) # Create all folders in the path
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = Path.joinpath(local_path, file_name) # Create folder for parent directory
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )


if __name__ == "__main__":
    # Download the file from s3 bucket
    client = boto3.client("s3")

    file_names, folders = get_file_folders(client, "latte2go-cafa-dataset")
    print(file_names, folders)

    # print('Downloading file from s3 bucket...')
    # download_file_from_s3(bucket_name='latte2go-cafa-dataset', s3_file_path="heterodata.pt",
    #                       data_dir="data/UniProt.InterPro.MULTISPECIES.DGG.parents/heterodata.pt")
    # print('Download to `data/UniProt.InterPro.MULTISPECIES.DGG.parents/heterodata.pt` complete!')