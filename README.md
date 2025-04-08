# Kidney-Disease-Classification-MLFlow-DVC
This project aims in predicting Kidney Diseases achieving 98% recall. 

**About Data**

There were around 5000+ images of normal kidney ctscan and 2000 images of tumor CT scan of Coronal and Axial cuts. 
Link- https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone

**Data Preprocessing**

Applied otsu' binarization. Used watershed segmentation(red line in below image) technique from OpenCV to segment the overlapping areas in the CT scan. Normalised the images. Image size taken was 224*224*3.

 ![image](https://github.com/user-attachments/assets/e7348c11-5dfa-48b7-811a-d9bcd9cabbd7)

Data augmentation techniques were used to handle imbalance data i.e. to increase more images for tumor class after train test split to  

**Model Used**

Used VGG-16 from Tensorflow/Keras. Used th concept of transfer learning. The base models paramters were kept freezed during training. The weights used are trained from IMAGENET. 3 fully connected layers were applied on top of it.

![image](https://github.com/user-attachments/assets/b0107c02-7d19-4b2e-8e03-39fd3b7f1040)

Train- Valdidation split - 80:20

Achieved almost 99% recall. Trained for 30 epochs with early stopping of 6 and batch size of 128.

![image](https://github.com/user-attachments/assets/c058c966-bdd0-4893-ae33-a81aa0aa10ea)

![image](https://github.com/user-attachments/assets/af9ad734-4e27-45ac-9081-1a03c8126fe2)

![image](https://github.com/user-attachments/assets/163b5104-5c47-4c56-927d-6e7a583a44f6)


**Techstacks**

![image](https://github.com/user-attachments/assets/dbba3ad3-e7ed-4790-b8cf-e4aeb5143170)

![image](https://github.com/user-attachments/assets/b461998f-1d77-4e36-912c-4c244f0dc8a6)

![image](https://github.com/user-attachments/assets/6518880a-5c54-48e0-b22a-743d5f90b1b1)

![image](https://github.com/user-attachments/assets/8ed9de77-d54f-46c8-b4f6-3e10eecd80bd)

![image](https://github.com/user-attachments/assets/af2fdc25-5f9a-4a47-b90d-4f48583b24e3)

![image](https://github.com/user-attachments/assets/88727cff-688b-4ae9-ba61-374fcaf8e1fe)

![image](https://github.com/user-attachments/assets/e2de6ca8-89bd-4de6-86ab-72708e46b80f)

![image](https://github.com/user-attachments/assets/a7151de5-f5c0-4ddc-9017-484d3443ca10)

![image](https://github.com/user-attachments/assets/f1a82f49-99c5-4fc6-bfdf-39f4b2ad2b12)

![image](https://github.com/user-attachments/assets/db02a2ce-b146-4609-91c3-6a3135c3fb81)

# Frontend And Results:-
**Note -This is just a prototype to get idea about my approach. Actual implementation was bit different in production in terms of frontend.**

![image](https://github.com/user-attachments/assets/171093a7-cc23-4f61-8500-436f023d0353)


![image](https://github.com/user-attachments/assets/433ec9d0-1b24-48de-99d3-a2a531e99a82)


![image](https://github.com/user-attachments/assets/9f776fb2-0d6e-41e5-9783-2ce526691082)









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
