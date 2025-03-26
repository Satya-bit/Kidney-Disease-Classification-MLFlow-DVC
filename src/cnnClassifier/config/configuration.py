# Configuration manager - This is where the data ingestion paths are prepared using the config(template of artifacts) and entity files(ensuring the return types)

from src.cnnClassifier.constants import *  #IMPORTING THE TWO CONSTANTS PATH OF CONFIG AND PARAMS FROM CONSTANTS

from src.cnnClassifier.utils.common import read_yaml, create_directories #Importing from utils the common functions like read_yaml and creat_directories.

from src.cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root]) #Dot accessing can be done because of Configbox which is the return type of the read_yaml function
        #This will retrurn artifacts the value of artifacts_root from config.yaml 

    def get_data_ingestion_config(self) -> DataIngestionConfig: #We have keep the return type as DataIngestionConfig beacuse the function does not return any other return type written in data_ingestion.
        #For example it will only return the 4 things mentioned in the class. It will return error if anything passed except this.
        #Also remeber we have used the ensure annotation decorators, So what will it do that it will not allow any other return type.
        #If let say the type of URL is str and if it is not str then it will give error. So be sure to check the datatype in config.yaml file and the types passed in entity.
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
            config = self.config.prepare_base_model
            
            create_directories([config.root_dir])

            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path=Path(config.base_model_path),
                updated_base_model_path=Path(config.updated_base_model_path),
                params_image_size=self.params.IMAGE_SIZE,
                params_learning_rate=self.params.LEARNING_RATE,
                params_include_top=self.params.INCLUDE_TOP,
                params_weights=self.params.WEIGHTS,
                params_classes=self.params.CLASSES
            )

            return prepare_base_model_config