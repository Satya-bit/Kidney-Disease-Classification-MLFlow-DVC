import os
import cv2 as cv
import numpy as np
import python_splitter
import shutil
import random
import imgaug.augmenters as iaa
from src.cnnClassifier.entity.config_entity import DataTransformationConfig 
 

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.data = self.config.data
        self.split = self.config.split
        self.train = self.config.train
        self.test = self.config.test

    @staticmethod
    def watershed(image_path):
        img = cv.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image at {image_path}")
            return None

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
        img[markers == -1] = [255, 0, 0]
        return img

    @staticmethod
    def augment_and_save_images(folder_path):
        """Augments all images in the specified folder"""
        image_files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("No images found in the specified folder.")
            return

        print(f"Found {len(image_files)} images. Augmenting...")

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = cv.imread(image_path)
            if image is None:
                print(f"Error: Unable to read {image_file}")
                continue

            augmenters = [
                iaa.Affine(rotate=(-15, 15)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.2),
                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
                iaa.Affine(shear=(-8, 8)),
                iaa.Affine(scale=(0.9, 1.1)),
                iaa.Multiply((0.8, 1.2)),
                iaa.LinearContrast((0.8, 1.2)),
                iaa.GaussianBlur(sigma=(0, 0.5))
            ]

            for i in range(1):  # Create 3 augmented versions
                selected = random.sample(augmenters, random.randint(3, 5))
                seq = iaa.Sequential(selected)
                augmented = seq.augment_image(image)
                aug_path = os.path.join(folder_path, f"aug_{i}_{image_file}")
                cv.imwrite(aug_path, augmented)
                print(f"Saved: {aug_path}")

        print("Augmentation complete.")

    def process_and_save_images(self):
        src_dir = str(self.data)
        dest_dir = str(self.config.root_dir)
        
        # Create destination folders
        normal_dir = os.path.join(dest_dir, 'Normal')
        tumor_dir = os.path.join(dest_dir, 'Tumor')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(tumor_dir, exist_ok=True)

        # Process and save original images
        for folder_name in ['Normal', 'Tumor']:
            folder_path = os.path.join(src_dir, folder_name)
            output_folder = normal_dir if folder_name == 'Normal' else tumor_dir

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    processed_img = DataTransformation.watershed(file_path)
                    if processed_img is not None:
                        output_path = os.path.join(output_folder, filename)
                        cv.imwrite(output_path, processed_img)
                        print(f"Processed and saved: {output_path}")

        # Perform train-test split
        python_splitter.split_from_folder(dest_dir, train=self.train, test=self.test)
        
        # Move the split folder if needed
        if os.path.exists('Train_Test_Folder'):
            shutil.move('Train_Test_Folder', str(self.split))
            
        # Augment tumor training images (after split)
        tumor_train_path = os.path.join(str(self.split), 'train', 'Tumor')
        if os.path.exists(tumor_train_path):
            DataTransformation.augment_and_save_images(tumor_train_path) #Accessing the static method wit calling the class name