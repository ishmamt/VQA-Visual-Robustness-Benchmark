"""
Utilit functions.
================================================
ishmamt
================================================
"""

import cv2
import os


def loadImage(imageDirectory, imageName):
    '''
    Returns the image from the given path.
    
        Parameters:
            imageDirectory (string): Image directory.
            imageName (string): Name of the image.
        
        Returns:
            image (numpy array): The image specified.
    '''
    
    return cv2.imread(os.path.join(imageDirectory, imageName))