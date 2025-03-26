# Components- It is responsible for the actual data ingestion at the specified locations in artifacts. Similarly for other components 
#It has all the main functions of this components
import os
import zipfile
import gdown
from src.cnnClassifier.logging.logs import logger
from src.cnnClassifier.utils.common import get_size

from src.cnnClassifier.entity.config_entity import DataIngestionConfig 

class DataIngestion:
    def __init__(self, config: DataIngestionConfig): #This comes from Configuration manager. 
        self.config = config #By this way we can access all the four variables of DataIngestionConfig


    
    def download_file(self)-> str:  # Downloading the data from google drive
       '''
       Fetch data from URL
       '''
       try:
           dataset_url = self.config.source_URL
           zip_download_dir= self.config.local_data_file
           os.makedirs("artifacts/data_ingestion", exist_ok=True)
           logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
           
           file_id=dataset_url.split("/")[-2]
           prefix='https://drive.google.com/uc?export=download&id='
           gdown.download(prefix+file_id,zip_download_dir) #Src and destination
           logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
       except Exception as e:
           raise e
       
    def extract_zip_file(self): #Extracting the zip file
           """
           zip_file_path: str
           Extracts the zip file into the data directory
           Function returns None
           """
           unzip_path = self.config.unzip_dir
           os.makedirs(unzip_path, exist_ok=True)
           with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref: #Take the source path form local_data_file and unzips at the destination path unzip_path
               zip_ref.extractall(unzip_path)
       
       
           
        