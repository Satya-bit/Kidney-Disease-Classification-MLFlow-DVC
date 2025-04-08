import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2 as cv
import os


class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename
    
    @staticmethod
    def watershed(img): #Applying watershed to input image by user. Note we can also use watreshed function from datatransformation but it was taking filepath as input but we need image as input. So we write the function again 
        """
        Applies watershed segmentation to an input image (numpy array).
        Returns the segmented image **without resizing**.
        """
        if img is None:
            raise ValueError("Input image is None")

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.001 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        return img  # Returns the segmented image
    
    def predict(self):
       
        model = load_model(os.path.join('artifacts','training','model.h5'))
        
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image=test_image.astype(np.uint8) #To apply uint8 it should be uint8

        test_image= PredictionPipeline.watershed(test_image)
       
        test_image = preprocess_input(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1) #Array of indices with max value So our softmax looks like [0.3,0.7]. Argmax returns the inidce with highest probaility in Softmax
        #This means argmax will return 1 because 0.7 is on indice 1. Here 1 means class 1 has highest probability. If we use result[1] this will give error since there is only one image we are giving as input.
        # If we give multiple images to predict and if we write result[1] this will give the argmax of image 1
        print(result)
        
        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        