"""
ViLT model for VQA.
================================================
ishmamt
================================================
"""

import torch
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering


class ViLT():
    '''
    Class for ViLT model.
    
        Attributes:
            modelName (string): The name of the model.
            processor (ViltProcessor):  The preprocessor for ViLT model.
            model (ViltForQuestionAnswering): The ViLT model.
    '''
    
    def __init__(self, modelName):
        '''
        Constructor method for ViLT class.
        
            Parameters:
                modelName (string): The name of the model.
                
        '''
        self.modelName = modelName
        self.processor, self.model = self.loadViLT()
        
    
    def loadViLT(self):
        '''
        Returns the preprocessor and ViLT model.
        
            Returns:
                processor (ViltProcessor):  The preprocessor for ViLT model.
                model (ViltForQuestionAnswering): The ViLT model.
        '''
        processor = ViltProcessor.from_pretrained(self.modelName)
        model = ViltForQuestionAnswering.from_pretrained(self.modelName)
        
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
        encoding = self.processor(image, str(question), return_tensors="pt")
        logits = self.model(**encoding).logits
        idx = torch.sigmoid(logits).argmax(-1).item()
        
        return self.model.config.id2label[idx]