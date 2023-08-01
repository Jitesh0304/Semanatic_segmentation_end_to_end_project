import os
import zipfile
from src.imageSegmentation import logger
from src.imageSegmentation.utils.common import get_size
import boto3
from src.imageSegmentation.entity.config_entity import DataIngestionConfig
from pathlib import Path




class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

        
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            try:
                s3_client = boto3.client('s3',
                                            aws_access_key_id = self.config.aws_access_id,
                                            aws_secret_access_key = self.config.aws_secrete_key)
                s3_client.download_file(Bucket= self.config.bucket_name, Key= self.config.data_filename, Filename = self.config.local_data_file)
            except Exception as e:
                raise e
            logger.info("download! with following info")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  


    
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)