from models.blip import BLIP
from imageio import imread
from utils import Logger

image = imread("test.jpg")
question = "What color is the alarm clock?"

logger = Logger(".")
model = BLIP(logger)

model.predict(image, question)