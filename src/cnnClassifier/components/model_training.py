import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
# tf.config.run_functions_eagerly(True)  # Force eager execution globally
import math
from src.cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

class Training:
    def __init__(self,config: TrainingConfig):
        self.config = config
    
    def get_base_model(self): #To get the updted model
        self.model=tf.keras.models.load_model(
            self.config.updated_base_model_path
            # self.config.base_model_path
        )
        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            metrics=["accuracy"]
        )
        
    def train_valid_generator(self): #Function for splitting the train and validation data with preprocessing

        datagenerator_kwargs = dict(
            # rescale = 1./255, #Common normslisation step
            # validation_split=0.20
            preprocessing_function=preprocess_input

        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], #Resizing the image to 224x224
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"  #A technique used for resizing the image. It uses a weighted average of the 4 nearest pixels.
        )

#preparing the validation data
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        ) #This will take 10% from normal and 10% from tumor directories roughly

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.validation_data,
            shuffle=False,
            **dataflow_kwargs
        ) #Resized image of 224x224 in validation set(20%, 10 % from each class)

#preparing the training data
        # if self.config.params_is_augmentation: #If augmentation is kept true it will try to transform the image. Note it won't be adding data to the training set. Just each batch will get randomly transformed as per the arguments passed
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,  # Wider rotation range
            vertical_flip=True,
            horizontal_flip=True,
            width_shift_range=0.1,  # Slightly larger shift range
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.3,  # Increased zoom range
            brightness_range=[0.3, 1.7],  # Increased brightness range
            **datagenerator_kwargs
        )
        # else:
        #     train_datagenerator = valid_datagenerator #If augemnetation is kept false it will not try to transform the image. Just rescaling. 

#Taking 80% of data from normal and tumor directories roughly. Doing data augmentation. Also shuffling to avoid memorization while training
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            shuffle=True,
            **dataflow_kwargs
        )
        
        
    @staticmethod #Saving the train model
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


#Iteration*batch=data points*epochs

    
    def train(self): #Function to train the model
        # self.steps_per_epoch = math.ceil(self.train_generator.samples / self.train_generator.batch_size) #This is number of iteration per epoch. The number of times the model parameters will be updated per epoch.
        # self.validation_steps = math.ceil(self.valid_generator.samples / self.valid_generator.batch_size) #After every iteration the model will be validated. 
        #For example if the validation has 100 samples and the batch size is 10 the evaluation will be done for 100/10=10 samples. A record of correct samples is kept. in these 10 samples. then afterwards its added together in the end.
        early_stopping = EarlyStopping(
            patience=self.config.params_patience,
            monitor=self.config.params_monitor
        )
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            # steps_per_epoch=self.steps_per_epoch,
            # validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=[early_stopping]
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )