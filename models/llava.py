"""
LLaVa 1.5 13B model for VQA.
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

import textwrap

import requests
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class LLaVa():
    '''
    Class for LLaVa model.
    
        Attributes:
            name (string): Simple name of the model.
            modelPath (string): The path or URL of the model.
            model (LLaVa): The LLaVa model.
            logger (Logger): Logger object.
    '''
    
    def __init__(self, logger, modelPath="4bit/llava-v1.5-13b-3GB", imageSize=336):
        '''
        Constructor method for LLaVa class.
        
            Parameters:
                modelPath (string): The name of the model.
                logger (Logger): Logger object.
                modelURL (string): The URL to download the model if it doesn't exist.
                imageSize (int): The size of the image pushed into the model.

        '''
        disable_torch_init()
        self.name = "LLaVa"
        self.modelPath = modelPath
        self.logger = logger
        self.imageSize = imageSize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer, self.model, self.image_processor, self.context_len = self.loadLLaVa()

    
    def loadLLaVa(self):
        '''
        Returns the LLaVa model.
        
            Returns:
                model (LLaVa): The LLaVa model.
        '''
        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                                                    model_path=self.modelPath, 
                                                    model_base=None, 
                                                    model_name=get_model_name_from_path(self.modelPath), 
                                                    load_4bit=True)
            model.eval()
            model = model.to(self.device)

        except exception as e:
            self.logger.error(f"{e} occured while loading model: {self.name}.")
            raise Exception(f"Unable to load model due to {e}.")
        
        return tokenizer, model, image_processor, context_len

    
    def preprocess(self, image):
        '''
        Preprocessing the image as needed.

            Parameters:
                image (numpy array): The image.

            Returns:
                image (PIL Image): The transformed image.
        '''
        image = Image.fromarray(image)

        args = {"image_aspect_ratio": "pad"}
        image_tensor = process_images([image], self.image_processor, args)

        return image_tensor.to(self.device, dtype=torch.float16)

    
    def create_prompt(self, prompt):
        CONV_MODE = "llava_v0"
        conv = conv_templates[CONV_MODE].copy()
        roles = conv.roles
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(roles[0], prompt)
        conv.append_message(roles[1], None)

        return conv.get_prompt(), conv


    def ask_image(self, image, prompt, conv):
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            keywords=[stop_str], tokenizer=self.tokenizer, input_ids=input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image,
                do_sample=True,
                temperature=0.01,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
    
    
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
            prompt, conv = create_prompt()  ## Add prompt with question
            answer = self.ask_image(image, prompt, conv)
            
        except exception as e:
            self.logger.error(f"{e} occured during prediction.")
            raise Exception(f"{e} occured during prediction.")
        
        return answer
