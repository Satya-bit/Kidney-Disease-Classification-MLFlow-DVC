# NOTE- To connect with MLFLOW run the code in research/model_evalutaion.ipynb one cell
#This is run file
from src.cnnClassifier.logging.logs import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.cnnClassifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_05_model_evaluation import EvaluationPipeline

# STAGE_NAME = "Data Ingestion stage"

# try:
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#         data_ingestion = DataIngestionTrainingPipeline() #Creates an instance of the class
#         data_ingestion.main() #starts the data ingestion pipeline
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

# STAGE_NAME = "Data Transformation stage"

# try:
#         logger.info(f"*******************")
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = DataTransformationPipeline()
#         obj.main()
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

# STAGE_NAME = "Prepare base model"

# try:
#         logger.info(f"*******************")
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         prepare_base_model = PrepareBaseModelTrainingPipeline()
#         prepare_base_model.main()
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e
    

# STAGE_NAME = "Training"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    model_trainer = ModelTrainingPipeline()
#    model_trainer.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e

# if __name__ == '__main__':
