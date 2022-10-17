"""
Code to create custom datasets. Inherits from
PyTorch Dataset class.
================================================
ishmamt
================================================
"""

from torch.utils.data import Dataset
import os
import errno
import json

from utils import loadImage


class VQADataset(Dataset):
    '''
    Dataset class for the VQA2.0 dataset. For more information please visit (https://github.com/GT-Vision-Lab/VQA).

        Attributes:
            name (string): Name of the dataset type (train/val/test).
            questionsJSON (string): Path to JSON file for the questions.
            annotationsJSON (string): Path to JSON file for the annotations.
            imageDirectory (string): Image directory.
            imagePrefix (string): Prefix of image names i.e. "COCO_train2014_".
    '''

    def __init__(self, name, questionsJSON, annotationsJSON, imageDirectory, imagePrefix):
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

        # Loading the annotations and questions from the JSON files
        self.annotations = json.load(open(self.annotationsJSON, 'r'))
        self.questions = json.load(open(self.questionsJSON, 'r'))

        # Preprocessing the dataset
        self.imageIds, self.imageNames = self.getImageIdsAndNames()
        self.ImageQuestionDictionary = self.getImageQuestionDictionary()
        self.questionIds = self.getQuestionIds()
        self.answersDictionary = self.getAnswersDictionary()
        self.questionsDictionary = self.getQuestionsDictionary()


    def __len__(self):
        '''
        Returns the length of the VQA2.0 dataset.

            Returns:
                datasetLenght (int): Length of the dataset.
        '''
        return len(self.questionsDictionary)


    def __getitem__(self, index):
        '''
        Returns an item from the VQA2.0 dataset given an index.

            Parameters:
                index (int): Index of the itam from the VQA2.0 dataset.
            Returns:
                item (tuple): Tuple containing the image, questions and annotations for the given index such as (image, question, answer, imageId, questionId)
        '''
        questionId = self.questionIds[index]
        imageId = self.ImageQuestionDictionary[questionId]
        image = loadImage(self.imageDirectory, self.imageNames[imageId])

        question = self.questionsDictionary[questionId]
        answer = self.answersDictionary[questionId]

        return image, question, answer, imageId, questionId


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


    def getImageQuestionDictionary(self):
        '''
        Returns a dictionary containing question IDs and corresponding image IDs.

            Returns:
                ImageQuestionDictionary (dictionary): Dictionary containing question IDs and corresponding image IDs such as {questionID: imageID}.
        '''
        ImageQuestionDictionary = dict()

        for question in self.questions["questions"]:
            ImageQuestionDictionary[int(question["question_id"])] = int(question["image_id"])

        return ImageQuestionDictionary


    def getQuestionIds(self):
        '''
        Returns a list containing question IDs.

            Returns:
                questionIds (list): A list containing question IDs.
        '''
        questionIds = list()

        for question in self.questions["questions"]:
            questionIds.append(int(question["question_id"]))

        return questionIds


    def getAnswersDictionary(self):
        '''
        Returns a dictionary containing question IDs and corresponding answers.

            Returns:
                answersDictionary (dictionary): Dictionary containing question IDs and corresponding answers such as {questionId: [answers]}.
        '''
        answersDictionary = dict()

        for annotation in self.annotations["annotations"]:  # list of dictionaries
            answersDictionary[int(annotation["question_id"])] = annotation["multiple_choice_answer"]

        return answersDictionary


    def getQuestionsDictionary(self):
        '''
        Returns a dictionary containing question IDs and corresponding questions.

            Returns:
                questionsDictionary (dictionary): Dictionary containing question IDs and corresponding question such as {questionId: question}.
        '''
        questionsDictionary = dict()

        for question in self.questions["questions"]:
            questionsDictionary[int(question["question_id"])] = question["question"]

        return questionsDictionary
