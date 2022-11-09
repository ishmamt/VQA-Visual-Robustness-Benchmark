"""
Code for generating the data dump as a report.
================================================
ishmamt
================================================
"""

import json
import os

class VQAReporter():
    '''
    Base class for generating the reports for VQA2.0 dataset.
    
        Attribute:
            name (string)
    '''
    
    def __init__(self, modelName, imageDirectory, outputDirectory, logger):
        '''
        Constructor method for the Reporter class.
        '''
        transformationName = os.path.normpath(imageDirectory).split(os.sep)[-1]
        self.name = f"{modelName}_{transformationName}.json"
        self.outputDirectory = outputDirectory
        self.logger = logger
        
        if not os.path.exists(outputDirectory):
            self.logger.info(f"Results output directory does not exist. Creating output directory: {outputDirectory}")
            os.makedirs(outputDirectory)
            
        self.report = {"dataset": "VQA2.0", "data": [], "accuracy": None, "totalAnswered": 0}
        
        
    def saveReport(self):
        self.logger.info(f"Saving report at {os.path.join(self.outputDirectory, self.name)}.")
        with open(os.path.join(self.outputDirectory, self.name), "w") as outfile:
            json.dump(self.report, outfile)
            
            
    def addToReport(self, correct, imageID, questionID, question, questionType, answer, predicted, accuracy, totalAnswered):
        entryDictionary = {
                           "correct": correct, 
                           "imageID": imageID,
                           "questionID": questionID,
                           "question": question,
                           "questionType": questionType,
                           "answer": answer,
                           "predicted": predicted
                          }
        self.report["data"].append(entryDictionary)
        self.report["accuracy"] = round(accuracy, 5)
        self.report["totalAnswered"] = totalAnswered