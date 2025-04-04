from src.cnnClassifier.logging.logs import logger

from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig

from pathlib import Path

import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf

class PrepareBaseModel:
    def __init__(self,config: PrepareBaseModelConfig):
        self.config= config
    
    def get_base_model(self): #Getting the base model VGG 16 form tensorflow and saving it in base_model_path
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        fc1=tf.keras.layers.Dense(
            1024,
            activation='relu'
        )(flatten_in)
        
        fc2=tf.keras.layers.Dense(
            512,
            activation='relu'
        )(fc1)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(fc2) #(flatten_in)This acts as an input to the final layer. This acts like sequential

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction  
        )
        
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    #Note actual VGG16 Model has has flatten+FC1+FC2+Output. In our model we only have flatten+Output. There are no weights between the flatten input and last conv layer. But still teh weights are updated. 
        
    def update_base_model(self): #This acts as a parameters passed in the _prepare_full_model
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model) #Saving the update base model
        
    #Staic method is like an independent function inside the class which does not need to pass self as an argument in the function.
    @staticmethod #Static because the save model doesnot depend on any instance variable.It does not need access to self, so making it a @staticmethod keeps it independent.
    def save_model(path: Path, model: tf.keras.Model): #Input type is Path and the return type for the model is tf.keras.Model
        model.save(path) 
