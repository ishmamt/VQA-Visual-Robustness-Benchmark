# The Generator module. Given a dataset of images,
# it can apply specified transformations.

# All transformation functions will also be
# implemented here.
# ================================================
# ishmamt
# ================================================

import cv2


class Generator():
    '''
    Generator class for applying transformations to images from a given dataset.

        Attributes:
                dataset: torch.dataset
                    The specified dataset to apply transformations.

    '''

    def __init__(self):
        self.dataset = None

    def transformToGrayscale(self, idx):
        '''
        Transforms an image to grayscale given an ID.

                Parameters:
                        idx (int): Image ID

                Returns:
                        grayImage (numpy array): Grayscale image
        '''
        image = self.dataset[idx]
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)

    def transformToGrayscaleInverted(self, idx):
        '''
        Transforms an image to grayscale and then inverts colors given an ID.

                Parameters:
                        idx (int): Image ID

                Returns:
                        invertedGrayImage (numpy array): Inverted grayscale image
        '''
        image = self.dataset[idx]
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invertedGrayImage = 255.0 - grayImage

        return cv2.cvtColor(invertedGrayImage, cv2.COLOR_GRAY2BGR)
