import logging
import os

import boto3
import tqdm

ACCESS_KEY='AKIA3G6ZHBW4BIY6ZWUB'
SECRET_KEY = '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'

# Download a file from s3 bucket to the `data` directory without using credentials
def download_file_from_s3(bucket_name:str, s3_file_path:str, data_dir="data", ):
    s3 = boto3.client('s3', region_name='us-west-2',
                      aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    s3.download_file(bucket_name, s3_file_path, data_dir)


if __name__ == "__main__":
    # Download the file from s3 bucket
    print('Downloading file from s3 bucket...')
    download_file_from_s3(bucket_name='latte2go-cafa-dataset', s3_file_path="Protein.pickle",
                          data_dir="data/UniProt.InterPro.MULTISPECIES.DGG.parents/Protein.pickle")
    print('Download `data/UniProt.InterPro.MULTISPECIES.DGG.parents/Protein.pickle` complete!')