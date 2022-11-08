from generator import Generator
from models.vilt import ViLT
from dataset import VQADataset
from utils import Logger


# Important Data for windows
name = "val"
questionsJSON = r"..\Hierarchical Co-Attention\Data\VQA\val\questions\val_quest_3K.json"
annotationsJSON = r"..\Hierarchical Co-Attention\Data\VQA\val\annotations\val_ann_3K.json"
imageDirectory = r"..\Hierarchical Co-Attention\Data\VQA\val\images\val3K"
imagePrefix = None
outputPath = r"."
logPath = r"."

# Important Data for Linux (Colab)
# name = "val"
# annotationsJSON = "/content/drive/MyDrive/VQA/Hierarchical_Co-attention/Data/val/annotations/val_ann_3K.json"
# questionsJSON = "/content/drive/MyDrive/VQA/Hierarchical_Co-attention/Data/val/questions/val_quest_3K.json"
# imageDirectory = "/content/drive/MyDrive/VQA/Hierarchical_Co-attention/Data/val/images/val3K"
# imagePrefix = None
# outputPath = "."
# logPath = "."


# Creating a logger
logger = Logger(logPath)
logger.info("Starting experiment.")


# Transformation of dataset
dataset = VQADataset(name, questionsJSON, annotationsJSON, imageDirectory, imagePrefix, logger)
logger.info("VQA2.0 dataset loaded.")

generator = Generator(dataset, logger)
transformationsList = ["Zoom-Blur_L1", "Elastic_L2"]
generator.transform(transformationsList, outputPath=outputPath)


# # Loading a model
# modelName = "dandelin/vilt-b32-finetuned-vqa"
# vilt = ViLT(modelName, logger)


# # Computing accuracy
# for idx in range(0, 10):
#     image, questions, answers, _, _ = generator.dataset[idx]

#     for idx, question in enumerate(questions):
#         print(question)
#         print(answers[idx])
#         print(vilt.predict(image, question))
#         print("\n\n")
