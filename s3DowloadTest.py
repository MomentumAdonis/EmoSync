import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import sys
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id='AKIASK5MCSKPTZUKOIV2',
    aws_secret_access_key='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',
    region_name='eu-north-1'
)

def download_from_s3(bucket_name, s3_key, local_path):
    """Download a file from S3 to local_path."""
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")

# Example usage:
BUCKET_NAME = "emo-sync-data"
download_from_s3(BUCKET_NAME, "TRAINING DATA/valence_long_filtered.csv", "/tmp/arousal_long_filtered.csv")

arousal_df = pd.read_csv("/tmp/arousal_long_filtered.csv")
print(arousal_df.head())