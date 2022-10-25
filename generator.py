"""
The Generator module. Given a dataset of images,
it can apply specified transformations.

All transformation functions will also be
implemented here.
================================================
ishmamt
================================================
"""

import os
import cv2
import errno
from tqdm import tqdm

from dataset import VQADataset
from models.vilt import ViLT
from utils import saveImage


class Generator():
    '''
    Generator class for applying transformations to images from a given dataset.

        Attributes:
            dataset (Dataset): The specified dataset to apply transformations.

    '''

    def __init__(self, name, questionsJSON, annotationsJSON, imageDirectory, imagePrefix=None):
        '''
        Constructor for the Generator class.

            Parameters:
                name (string): Name of the dataset type (train/val/test).
                questionsJSON (string): Path to JSON file for the questions.
                annotationsJSON (string): Path to JSON file for the annotations.
                imageDirectory (string): Image directory.
                imagePrefix (string): Prefix of image names i.e. "COCO_train2014_".
        '''
        self.dataset = VQADataset(name, questionsJSON, annotationsJSON, imageDirectory, imagePrefix)
        self.validTransformations = ["Grayscale", "Grayscale-Inverse"]
        
    
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
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), outputPath)

        for transformation in transformationsList:
            # if transformation not in self.validTransformations:
            #     print("INVALID TRANSFORMATION")  # Put this in log
            #     continue

            pBar = tqdm(total=len(self.dataset))  # progress bar
            # Loop over all images in the dataset
            for idx in range(len(self.dataset)):
                pBar.update(1)
                transformedImage = transformation(idx)
                if saveOutputs:
                    imageId = self.dataset.imageIds[idx]
                    saveImage(transformedImage, outputPath, self.dataset.imageNames[imageId])


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

        return cv2.cvtColor(invertedGrayImage, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    name = "val"
    questionsJSON = r"..\Hierarchical Co-Attention\Data\VQA\val\questions\val_quest_3K.json"
    annotationsJSON = r"..\Hierarchical Co-Attention\Data\VQA\val\annotations\val_ann_3K.json"
    imageDirectory = r"..\Hierarchical Co-Attention\Data\VQA\val\images\val3K"
    outputPath = r"."

    # annotationsJSON = "/content/drive/MyDrive/VQA/Hierarchical_Co-attention/Data/val/annotations/val_ann_3K.json"
    # questionsJSON = "/content/drive/MyDrive/VQA/Hierarchical_Co-attention/Data/val/questions/val_quest_3K.json"
    # imageDirectory = "/content/drive/MyDrive/VQA/Hierarchical_Co-attention/Data/val/images/val3K"
    # outputPath = "."

    modelName = "dandelin/vilt-b32-finetuned-vqa"


    generator = Generator(name, questionsJSON, annotationsJSON, imageDirectory)
    transformationsList = [generator.transformToGrayscale]
    # vilt = ViLT(modelName=modelName)

    # for idx in range(0, 10):
    #     image, questions, answers, _, _ = generator.dataset[idx]

    #     for idx, question in enumerate(questions):
    #         print(question)
    #         print(answers[idx])
    #         print(vilt.predict(image, question))
    #         print("\n\n")

    generator.transform(transformationsList, outputPath=outputPath)
