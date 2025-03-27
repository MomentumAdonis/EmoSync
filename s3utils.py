import os
import boto3
from dotenv import load_dotenv
load_dotenv()


def get_s3_client():
    """
    Creates and returns an S3 client using credentials and region information
    from environment variables:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_REGION (defaults to 'us-east-1' if not provided)
    """
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region_name = os.environ.get('AWS_REGION', 'us-east-1')

    if not aws_access_key_id or not aws_secret_access_key:
        raise Exception("AWS credentials not set in environment variables.")
    
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )



def download_from_s3(bucket_name, s3_key, local_path):
    """
    Downloads a file from the specified S3 bucket and key, saving it to local_path.
    
    Parameters:
      bucket_name (str): Name of the S3 bucket.
      s3_key (str): The key (path) to the file in S3.
      local_path (str): The local file path where the file will be saved.
    """
    client = get_s3_client()
    # Ensure the directory for local_path exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} from bucket {bucket_name} to {local_path}")



def upload_to_s3(bucket_name, s3_key, local_path):
    """
    Uploads a local file to the specified S3 bucket and key.
    
    Parameters:
      bucket_name (str): Name of the S3 bucket.
      s3_key (str): The destination key (path) in the S3 bucket.
      local_path (str): The local file path to upload.
    """
    client = get_s3_client()
    client.upload_file(local_path, bucket_name, s3_key)
    print(f"Uploaded {local_path} to {bucket_name}/{s3_key}")



def list_files_in_bucket(bucket_name, prefix=""):
    """
    Lists all file keys in the specified S3 bucket that start with the given prefix.
    
    Parameters:
      bucket_name (str): Name of the S3 bucket.
      prefix (str): (Optional) Prefix filter for the keys.
      
    Returns:
      List[str]: A list of file keys.
    """
    client = get_s3_client()
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents']]
    else:
        return []
