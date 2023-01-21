"""
Utilit functions.
Also implemented the logger class.
================================================
ishmamt
================================================
"""

import cv2
from imageio import imread
import os
from datetime import datetime


def loadImage(imageDirectory, imageName):
    '''
    Returns the image from the given path.
    
        Parameters:
            imageDirectory (string): Image directory.
            imageName (string): Name of the image.
        
        Returns:
            image (numpy array): The image specified.
    '''
    image = imread(os.path.join(imageDirectory, imageName))

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def saveImage(image, imageDirectory, imageName):
    '''
    Saves an image in the given path.
    
        Parameters:
            image (numpy array): Image to be saved.
            imageDirectory (string): Image directory.
            imageName (string): Name of the image.
    '''
    cv2.imwrite(os.path.join(imageDirectory, imageName), image)
    

class Logger():
    '''
    Class to handle logging.

        Attributes:
            logPath (str): Path to generate the log file.
            importanceLevels (list): List of importance levels. They are: DEBUG, INFO, WARNING, ERROR, CRITICAL
            datetimeFormat (str): Datetime format string. By default they are: Day-Month-Year  Hour:Minute:Second.
    '''

    def __init__(self, logPath):
        '''
        Constructor method to intialize a logger class.

        Parameters:
            logPath (str): Path to generate the log file.

        Returns:
            logger (logger object): The logger object.
        '''
        if not os.path.exists(logPath):
            print(f"Log directory does not exist. Creating log directory: {logPath}")
            os.makedirs(logPath)

        self.logPath = logPath
        self.importanceLevels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.datetimeFormat = "%d-%m-%Y  %H:%M:%S"


    def configureLogMessage(self, level, message):
        '''
        Configure a message to add to log file.

            Parameters:
                level (string): Importance level of the message. Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
                message (string): Message to add to the log file.

            Returns:
                configuredMessage (string): Message configured to specification.
        '''
        if level not in self.importanceLevels:
            raise Exception(f"Invalid importance level of log message: {level}. It should be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")

        return f"{level} | {datetime.now().strftime(self.datetimeFormat)} | {str(message)}"

    
    def writeToLog(self, message):
        '''
        Method to open and append a message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        with open(os.path.join(self.logPath, "experiment.log"), "a") as logFile:
            logFile.write(f"{message}\n")

    
    def debug(self, message):
        '''
        Method for adding a debug message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.writeToLog(self.configureLogMessage("DEBUG", message))

    
    def info(self, message):
        '''
        Method for adding a info message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.writeToLog(self.configureLogMessage("INFO", message))

    
    def warning(self, message):
        '''
        Method for adding a warning message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.writeToLog(self.configureLogMessage("WARNING", message))

    
    def error(self, message):
        '''
        Method for adding a error message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.writeToLog(self.configureLogMessage("ERROR", message))


    def critical(self, message):
        '''
        Method for adding a critical message to the log file.

            Parameters:
                message (string): Message to be addded to the log file.
        '''
        self.writeToLog(self.configureLogMessage("CRITICAL", message))