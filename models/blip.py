"""
BLIP model for VQA.
================================================
ishmamt
================================================
"""

from logging import exception
import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from BLIP.models.blip_vqa import blip_vqa


class BLIP():
    '''
    Class for BLIP model.
    
        Attributes:
            name (string): Simple name of the model.
            modelPath (string): The path or URL of the model.
            model (blip_vqa): The BLIP model.
            logger (Logger): Logger object.
    '''
    
    def __init__(self, logger, modelPath="", 
                modelURL="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth", imageSize=480):
        '''
        Constructor method for BLIP class.
        
            Parameters:
                modelPath (string): The name of the model.
                logger (Logger): Logger object.
                modelURL (string): The URL to download the model if it doesn't exist.
                imageSize (int): The size of the image pushed into the model.

        '''
        self.name = "BLIP"
        self.modelPath = modelPath
        self.logger = logger
        self.modelURL = modelURL
        self.imageSize = imageSize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.loadBLIP()
        
    
    def loadBLIP(self):
        '''
        Returns the BLIP model.
        
            Returns:
                model (blip_vqa): The BLIP model.
        '''
        try:
            if not os.path.exists(self.modelPath):
                model = blip_vqa(pretrained=self.modelURL, image_size=self.imageSize, vit='base')
                self.logger.info(f"BLIP model downloaded using {self.modelURL}.")
            else:
                model = blip_vqa(pretrained=self.modelPath, image_size=self.imageSize, vit='base')
                self.logger.info(f"BLIP model loaded using {self.modelPath}.")
            
            model.eval()
            model = model.to(self.device)
        except exception as e:
            self.logger.error(f"{e} occured while loading model: {self.name}.")
            raise Exception(f"Unable to load model due to {e}.")
        
        return model

    
    def preprocess(self, image):
        '''
        Preprocessing the image as needed.

            Parameters:
                image (numpy array): The image.

            Returns:
                image (PIL Image): The transformed image.
        '''
        image = Image.fromarray(image)
        transform = transforms.Compose([
                                        transforms.Resize((self.imageSize, self.imageSize),interpolation=InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                                    ])
        image = transform(image).unsqueeze(0).to(self.device)

        return image
    
    
    def predict(self, image, question):
        '''
        Predicts the answer given the an image and a question using the BLIP model.
        
            Parameters:
                image (numpy array): The image.
                question (string): The question.
                
            Returns:
                answer (string): The answer to the given question.
        '''
        try:
            image = self.preprocess(image)
            with torch.no_grad():
                answer = self.model(image, question, train=False, inference='generate') 
                answer = answer[0]
        except exception as e:
            self.logger.error(f"{e} occured during prediction.")
            raise Exception(f"{e} occured during prediction.")
        
        return answer