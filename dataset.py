"""
Code to create custom datasets. Inherits from
PyTorch Dataset class.

All transformation functions will also be
implemented here.
================================================
ishmamt
================================================
"""

from torch.utils.data import Dataset
import os
import errno


class VQADataset(Dataset):
    '''
    Dataset class for the VQA2.0 dataset. For more information please visit (https://github.com/GT-Vision-Lab/VQA).

        Attributes:
                dataset: torch.dataset
                    The specified dataset to apply transformations.

    '''
    
    def __init__(self, name, questionsJSON, annotationsJSON, imageDirectory, imagePrefix=None):
        '''
        Constructor for the VQADataset class.
        
                Parameters:
                        name (string): Name of the dataset type (train/val/test).
                        questionsJSON (string): Path to JSON file for the questions.
                        annotationsJSON (string): Path to JSON file for the annotations.
                        imageDirectory (string): Image directory.
                        imagePrefix (string): Prefix of image names i.e. "COCO_train2014_".
        '''
        self.name = name
        self.questionsJSON = questionsJSON
        self.annotationsJSON = annotationsJSON
        self.imageDirectory = imageDirectory
        self.imagePrefix = imagePrefix
        
        if self.imagePrefix is None:
            self.imagePrefix = f"COCO_{self.name}2014_"
        
        # Checks to see if the files and directories exist
        if not os.path.exists(self.annotationsJSON):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.annotationsJSON)
        if not os.path.exists(self.questionsJSON):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.questionsJSON)
        if not os.path.isdir(self.imageDirectory):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.imageDirectory)
        
        # Preprocessing the dataset
        self.imageIds, self.imageNames = self.getImageIdsAndNames()
        
        
    def __len__(self):
        '''
        Returns the length of the VQA2.0 dataset.
        
            Returns:
                datasetLenght (int): Length of the dataset.
        '''
        return len(self.imageIds)
    
    
    def __getitem__(self, index):
        '''
        Returns an item from the VQA2.0 dataset given an index.
        
            Parameters:
                index (int): Index of the itam from the VQA2.0 dataset.
            Returns:
                item (tuple): Tuple containing the image, questions and annotations for the given index.
        '''
        return None
    
    
    def getImageIdsAndNames(self):
        '''
        Returns a the image IDs and names.
        
            Returns:
                imageIds (list): List of the image IDs.
                imageNames (dictionary): Dictionary containing image names such as {imageId: imageName}.
        '''
        imageIds = list()
        imageNames = dict()

        for imageName in os.listdir(self.imageDirectory):
            id = imageName.split(".")[0].rpartition(self.imagePrefix)[-1]  # image name: COCO_train2014_000000000123.jpg
            imageIds.append(int(id))
            imageNames[int(id)] = imageName

        return imageIds, imageNames
    
    
    
if __name__ == "__main__":
    name = "val"
    questionsJSON = r"..\Hierarchical-Co-attention-VQA\Data\val\questions\val_quest_3K.json"
    annotationsJSON = r"..\Hierarchical-Co-attention-VQA\Data\val\annotations\val_ann_3K.json"
    imageDirectory = r"..\Hierarchical-Co-attention-VQA\Data\val\images\val3K"
    
    dataset = VQADataset(name, questionsJSON, annotationsJSON, imageDirectory)
    imageIds, imageNames = dataset.getImageIdsAndNames()
    
    print(imageIds[:11])
    print(imageNames[imageIds[0]])
    print(len(dataset))