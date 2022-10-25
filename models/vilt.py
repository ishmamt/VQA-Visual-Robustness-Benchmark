"""
ViLT model for VQA.
================================================
ishmamt
================================================
"""

from logging import exception
import torch
import errno
import os
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering


class ViLT():
    '''
    Class for ViLT model.
    
        Attributes:
            modelName (string): The name of the model.
            processor (ViltProcessor):  The preprocessor for ViLT model.
            model (ViltForQuestionAnswering): The ViLT model.
            logger (Logger): Logger object.
    '''
    
    def __init__(self, modelName, logger):
        '''
        Constructor method for ViLT class.
        
            Parameters:
                modelName (string): The name of the model.
                logger (Logger): Logger object.
                
        '''
        self.modelName = modelName
        self.logger = logger
        self.processor, self.model = self.loadViLT()
        
    
    def loadViLT(self):
        '''
        Returns the preprocessor and ViLT model.
        
            Returns:
                processor (ViltProcessor):  The preprocessor for ViLT model.
                model (ViltForQuestionAnswering): The ViLT model.
        '''
        try:
            processor = ViltProcessor.from_pretrained(self.modelName)
            model = ViltForQuestionAnswering.from_pretrained(self.modelName)
            self.logger.info(f"ViLT model loaded using {self.modelName}.")
        except exception as e:
            self.logger.error(f"{e} occured while loading model: {self.modelName}.")
            raise Exception(f"Unable to load model due to {e}.")
        
        return processor, model
    
    
    def predict(self, image, question):
        '''
        Predicts the answer given the an image and a question using the ViLT model.
        
            Parameters:
                image (numpy array): The image.
                question (string): The question.
                
            Returns:
                answer (String): The answer to the given question.
        '''
        try:
            encoding = self.processor(image, str(question), return_tensors="pt")
            logits = self.model(**encoding).logits
            idx = torch.sigmoid(logits).argmax(-1).item()
        except exception as e:
            self.logger.error(f"{e} occured during prediction.")
            raise Exception(f"{e} occured during prediction.")
        
        return self.model.config.id2label[idx]