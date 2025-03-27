import os
import boto3
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists.
load_dotenv()


def get_s3_client():
    """
    Creates and returns an S3 client using credentials and region information
    obtained from environment variables:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_REGION (defaults to 'us-east-1' if not provided)
      
    Raises an Exception if the required credentials are not found.
    """
    # Retrieve AWS credentials and region from environment variables.
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region_name = os.environ.get('AWS_REGION', 'us-east-1')

    # Ensure both AWS access key and secret key are provided.
    if not aws_access_key_id or not aws_secret_access_key:
        raise Exception("AWS credentials not set in environment variables.")
    
    # Return a boto3 S3 client initialized with the retrieved credentials and region.
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )


def download_from_s3(bucket_name, s3_key, local_path):
    """
    Downloads a file from a specified S3 bucket and key, saving it locally.
    
    Parameters:
      bucket_name (str): Name of the S3 bucket.
      s3_key (str): The S3 key (i.e., the path to the file within the bucket).
      local_path (str): Local file path where the file will be saved.
      
    This function also ensures that the destination directory exists.
    """
    # Get the S3 client using the helper function.
    client = get_s3_client()
    # Create local directories if they do not exist.
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    # Download the file from S3 to the specified local path.
    client.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} from bucket {bucket_name} to {local_path}")


def upload_to_s3(bucket_name, s3_key, local_path):
    """
    Uploads a local file to a specified S3 bucket and key.
    
    Parameters:
      bucket_name (str): Name of the S3 bucket.
      s3_key (str): The destination key (path) in the S3 bucket.
      local_path (str): The local file path to upload.
      
    After uploading, it prints a confirmation message.
    """
    # Get the S3 client.
    client = get_s3_client()
    # Upload the file from local_path to the specified bucket and key.
    client.upload_file(local_path, bucket_name, s3_key)
    print(f"Uploaded {local_path} to {bucket_name}/{s3_key}")


def list_files_in_bucket(bucket_name, prefix=""):
    """
    Lists all file keys in the specified S3 bucket that start with a given prefix.
    
    Parameters:
      bucket_name (str): Name of the S3 bucket.
      prefix (str): Optional prefix to filter the keys.
      
    Returns:
      List[str]: A list of file keys that match the prefix. Returns an empty list if no files are found.
    """
    # Get the S3 client.
    client = get_s3_client()
    # List objects in the bucket filtered by the prefix.
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    # Check if any objects were returned; if so, extract their keys.
    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents']]
    else:
        return []
