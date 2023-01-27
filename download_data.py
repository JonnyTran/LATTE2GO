import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import boto3
import botocore
import tqdm
from botocore.client import BaseClient


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


def download_files(s3: BaseClient, bucket_name:str, local_path:str, file_names:List[str], folders:List[str]):
    local_path = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(local_path, folder) # Create all folders in the path
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm.tqdm(file_names, desc=f"Downloading dataset files to {local_path}", total=len(file_names)):
        object_size = s3.head_object(Bucket=bucket_name, Key=file_name)["ContentLength"]

        with tqdm.tqdm(total=object_size, unit="B", unit_scale=True, desc=file_name) as pbar:
            file_path = Path.joinpath(local_path, file_name) # Create folder for parent directory
            file_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(
                bucket_name,
                file_name,
                str(file_path),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),

            )


if __name__ == "__main__":
    # Download the file from s3 bucket
    try:
        client = boto3.client("s3")
    except botocore.exceptions.NoCredentialsError:
        logging.error("No AWS credentials found! Please create your AWS credentials if you haven't done so, "
                      "and store them at ~/.aws/credentials.")
        exit(1)

    # Create an ArgumentParser with two parameters, bucket_name and output_dir to call the script from the command line
    parser = ArgumentParser()
    parser.add_argument("--bucket_name", type=str, help="Name of the bucket to download from", default="latte2go-cafa-dataset")
    parser.add_argument("--output_dir", type=str, help="Directory to download the files to", default="data/")
    args = parser.parse_args()

    file_names, folders = get_file_folders(client, args.bucket_name)

    download_files(
        client,
        args.bucket_name,
        args.output_dir,
        file_names,
        folders
    )




