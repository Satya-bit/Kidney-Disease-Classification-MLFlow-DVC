# Pipeline- To run the main functions of the components in specific order to avoid error.

from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.data_ingestion import DataIngestion
from src.cnnClassifier.logging.logs import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager() #Initializing the configuration manager class to get data ingestion config
        data_ingestion_config = config.get_data_ingestion_config() #Getting the data ingestion config for preparing the data ingestion paths
        data_ingestion = DataIngestion(config=data_ingestion_config) #Initializing the data ingestion class to access the data ingestion main functions
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        

if __name__ == '__main__': 
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline() #Creates an instance of the class
        obj.main() #starts the pipeline
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    
#We have written this main because what happens if we donot write this then suppose we import this stage_01_data_ingestion.py file 
#in another file then it will run all the functions in stage_01_data_ingestion.py file just by importing statements. We can run this pipeline when we run this file directly.
    #If you want to run this function we will call this file from main.py and define the below code there. We have written this below code to keep the track of code in pipeline.py
    # We are already running this already in main.py. But suppose we want to run each componenet sepearately then we can run this code here that's why we wrote this code.  
    #Remeber the pizza shop and the customer example