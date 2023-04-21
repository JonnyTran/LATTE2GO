import logging
import os
import traceback
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import boto3
import botocore
import tqdm
from botocore.client import BaseClient
import urllib.request

from logzero import logger
from pandas.io.common import is_url


def get_s3_file_folders(s3:BaseClient, bucket_name:str, prefix=""):
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

        response = s3.list_objects_v2(**updated_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders


def download_s3_files(s3: BaseClient, bucket_name:str, output_dir:str, file_names:List[str], folders:List[str]):
    output_dir = Path(output_dir)

    for folder in folders:
        folder_path = Path.joinpath(output_dir, folder) # Create all folders in the path
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm.tqdm(file_names, desc=f"Downloading dataset files to {output_dir}", total=len(file_names)):
        try:
            object_size = s3.head_object(Bucket=bucket_name, Key=file_name)["ContentLength"]
        except:
            object_size = None

        with tqdm.tqdm(total=object_size, unit="B", unit_scale=True, desc=file_name) as pbar:
            file_path = Path.joinpath(output_dir, file_name) # Create folder for parent directory
            file_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(
                bucket_name,
                file_name,
                str(file_path),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )


def download_url_files(output_dir:str, baseurl:str, files:List[str]):
    for filename in tqdm.tqdm(files, desc='Downloading DeepGraphGO dataset'):
        urlpath = os.path.join(baseurl, filename)
        urllib.request.urlretrieve(urlpath,
                                   filename=os.path.join(output_dir, filename))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bucket_name", type=str, help="Name of the bucket to download from",
                        default="latte2go-cafa-datasets")
    parser.add_argument("--output_dir", type=str, help="Directory to download the files to", default="data/")
    args = parser.parse_args()

    # Download the HeteroNetwork datasets
    try:
        client = boto3.client("s3")
        file_names, folders = get_s3_file_folders(client, args.bucket_name)
        download_s3_files(client, bucket_name=args.bucket_name, output_dir=args.output_dir,
                          file_names=file_names, folders=folders)

    except botocore.exceptions.NoCredentialsError as nce:
        logger.error("No AWS credentials found! Please create your AWS credentials if you haven't done so, "
                      "and store them at ~/.aws/credentials or run `aws configure`.")
        traceback.print_exc()
        exit(1)


    # Download DeepGraphGO dataset
    if not os.path.exists(os.path.join(args.output_dir, "DeepGraphGO")):
        download_url_files(output_dir=os.path.join(args.output_dir, "DeepGraphGO/data"),
                           baseurl="https://raw.githubusercontent.com/yourh/DeepGraphGO/master/data/",
                           files=['data.zip', 'data.z01', 'data.z02', 'data.z03', 'data.z04', 'data.z05',
                                            'data.z06'])
        print("Unzipping DeepGraphGO dataset")
        os.system("cd data/DeepGraphGO/data")
        os.system("dtrx -fo data.zip")
        print("Preprocessing DeepGraphGO dataset")
        os.system("python preprocessing.py ppi_mat.npz ppi_dgl_top_100")
        print("Done!")