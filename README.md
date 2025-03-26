# Kidney-Disease-Classification-MLFlow-DVC
This project aims in predicting Kidney Diseases.

## Workflows

1. Update config.yaml - Where we will write the structure of artifacts in key value pairs

2. Update secrets.yaml [Optional]

3. Update params.yaml - Will have training parameters used for Model Trainer

4. Update the entity - This is where we will write the return type of the functions like data ingestion, model trainer etc.(whether its str or 
Path). To access this as class variable from other files we use dataclass(entity)

5. Update the configuration manager in src config -  This is where the components(data ingestion, validation, model trainer) paths are prepared 
using the config(template of artifacts) and entity files(ensuring the return types)

6. Update the components- Responsible for actual data ingestion and other components. All the main function of components are written here.

7. Update the pipeline- To run the main functions of the components in specific order to avoid error.

8. Update the main.py

9. Update the dvc.yaml

10. app.py
