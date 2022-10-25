"""
The Generator module. Given a dataset of images,
it can apply specified transformations.

All transformation functions will also be
implemented here.
================================================
ishmamt
================================================
"""

from logging import exception
import os
import cv2
import errno
from tqdm import tqdm
import numpy as np

from utils import saveImage


class Generator():
    '''
    Generator class for applying transformations to images from a given dataset.

        Attributes:
            dataset (Dataset): The specified dataset to apply transformations.
            validTransformations (dictionary): The dictionary of valid transformations such that: {"transformationName": transformationMethod}
            logger (Logger): Logger object.
    '''

    def __init__(self, dataset, logger):
        '''
        Constructor for the Generator class.

            Parameters:
                dataset (Dataset): The specified dataset to apply transformations.
                logger (Logger): Logger object.
        '''
        self.dataset = dataset
        self.logger = logger
        self.validTransformations = {
                                    "Grayscale": self.transformToGrayscale, 
                                    "Grayscale-Inverse": self.transformToGrayscaleInverted
                                    }
        
    
    def transform(self, transformationsList, saveOutputs=True, outputPath="."):
        '''
        Method to transform the whole image dataset, given the specified transformations.
        
            Parameters:
                transformationsList (list): List of transformation methods to apply
                saveOutputs (boolean): True if the transformed dataset is to be saved.
                outputPath (string): Directory to save the transformed datasets.
        
            Returns:
                transformedDatasets (list): The transformed dataset.
        '''
        if saveOutputs:
            # Checks to see if the files and directories exist
            if not os.path.exists(outputPath):
                self.logger.error("Invalid outputPath to save images.")
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), outputPath)

        for transformation in transformationsList:
            if transformation not in self.validTransformations:
                self.logger.warning(f"Invalid transformation: {transformation}. It should be one of {self.validTransformations.keys()}")
                continue

            if not os.path.exists(os.path.join(outputPath, transformation)):
                os.makedirs(os.path.join(outputPath, transformation))
            
            pBar = tqdm(total=len(self.dataset))  # progress bar
            transformationMethod = self.validTransformations[transformation]  # getting the method
            self.logger.info(f"Starting the transformation: {transformation} over the dataset.")
            savedCounter = 0  # A counter to figure out how many images were succesfully transformed and saved.
            # Loop over all images in the dataset
            for idx in range(len(self.dataset)):
                pBar.update(1)
                try:
                    transformedImage = transformationMethod(idx)
                except exception as e:
                    self.logger.error(f"{e} occured when using {transformation} on image number: {idx}.")
                    continue
                
                if saveOutputs:
                    imageId = self.dataset.imageIds[idx]
                    try:
                        saveImage(transformedImage, os.path.join(outputPath, transformation), self.dataset.imageNames[imageId])
                        savedCounter += 1
                    except exception as e:
                        self.logger.error(f"Failed to save image number: {idx} because {e} occured.")
                        continue
            
            self.logger.info(f"Saved {savedCounter} images for transformation: {transformation}.")


    def transformToGrayscale(self, idx):
        '''
        Transforms an image to grayscale given an ID.

            Parameters:
                idx (int): Image ID

            Returns:
                grayImage (numpy array): Grayscale image
        '''
        image, _, _, _, _ = self.dataset[idx]
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)


    def transformToGrayscaleInverted(self, idx):
        '''
        Transforms an image to grayscale and then inverts colors given an ID.

            Parameters:
                idx (int): Image ID

            Returns:
                invertedGrayImage (numpy array): Inverted grayscale image
        '''
        image, _, _, _, _ = self.dataset[idx]
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invertedGrayImage = 255.0 - grayImage
        invertedGrayImage = np.float32(invertedGrayImage)

        return cv2.cvtColor(invertedGrayImage, cv2.COLOR_GRAY2BGR)
