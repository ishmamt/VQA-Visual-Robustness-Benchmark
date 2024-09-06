from models.llava import LLaVa
from imageio import imread
from utils import Logger

image = imread("test.jpg")
question = "What color is the alarm clock?"

logger = Logger(".")
model = LLaVa(logger)

model.predict(image, question)