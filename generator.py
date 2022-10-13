"""
The Generator module. Given a dataset of images,
it can apply specified transformations.

All transformation functions will also be
implemented here.
================================================
ishmamt
================================================
"""

from email import generator
import cv2

from dataset import VQADataset


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


    def transformToGrayscale(self, idx):
        '''
        Transforms an image to grayscale given an ID.

            Parameters:
                idx (int): Image ID

            Returns:
                grayImage (numpy array): Grayscale image
        '''
        image = self.dataset[idx]
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
        image = self.dataset[idx]
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invertedGrayImage = 255.0 - grayImage

        return cv2.cvtColor(invertedGrayImage, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    name = "val"
    questionsJSON = r"..\Hierarchical-Co-attention-VQA\Data\val\questions\val_quest_3K.json"
    annotationsJSON = r"..\Hierarchical-Co-attention-VQA\Data\val\annotations\val_ann_3K.json"
    imageDirectory = r"..\Hierarchical-Co-attention-VQA\Data\val\images\val3K"
    
    generator = Generator(name, questionsJSON, annotationsJSON, imageDirectory)