"""
The Generator module. Given a dataset of images,
it can apply specified transformations.

All transformation functions will also be
implemented here.
================================================
ishmamt
Nipun
================================================
"""

from logging import exception
import cv2
import errno
from tqdm import tqdm
import numpy as np
from imageio import imread
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import os
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor

from utils import saveImage


class Generator():
    '''
    Generator class for applying transformations to images from a given dataset.

        Attributes:
            dataset (Dataset): The specified dataset to apply transformations.
            validTransformations (dictionary): The dictionary of valid transformations such that: {"transformationName": transformationMethod}
            logger (Logger): Logger object.
    '''

    def __init__(self, dataset, logger):
        '''
        Constructor for the Generator class.

            Parameters:
                dataset (Dataset): The specified dataset to apply transformations.
                logger (Logger): Logger object.
        '''
        self.dataset = dataset
        self.logger = logger
        self.validTransformations = {
                                    "Grayscale": self.transformToGrayscale, 
                                    "Grayscale-Inverse": self.transformToGrayscaleInverted
                                    }
        
    
    def transform(self, transformationsList, saveOutputs=True, outputPath="."):
        '''
        Method to transform the whole image dataset, given the specified transformations.
        
            Parameters:
                transformationsList (list): List of transformation methods to apply
                saveOutputs (boolean): True if the transformed dataset is to be saved.
                outputPath (string): Directory to save the transformed datasets.
        
            Returns:
                transformedDatasets (list): The transformed dataset.
        '''
        if saveOutputs:
            # Checks to see if the files and directories exist
            if not os.path.exists(outputPath):
                self.logger.error("Invalid outputPath to save images.")
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), outputPath)

        for transformation in transformationsList:
            if transformation not in self.validTransformations:
                self.logger.warning(f"Invalid transformation: {transformation}. It should be one of {self.validTransformations.keys()}")
                continue

            if not os.path.exists(os.path.join(outputPath, transformation)):
                os.makedirs(os.path.join(outputPath, transformation))
            
            pBar = tqdm(total=len(self.dataset))  # progress bar
            transformationMethod = self.validTransformations[transformation]  # getting the method
            self.logger.info(f"Starting the transformation: {transformation} over the dataset.")
            savedCounter = 0  # A counter to figure out how many images were succesfully transformed and saved.
            # Loop over all images in the dataset
            for idx in range(len(self.dataset)):
                pBar.update(1)
                try:
                    transformedImage = transformationMethod(idx)
                except exception as e:
                    self.logger.error(f"{e} occured when using {transformation} on image number: {idx}.")
                    continue
                
                if saveOutputs:
                    imageId = self.dataset.imageIds[idx]
                    try:
                        saveImage(transformedImage, os.path.join(outputPath, transformation), self.dataset.imageNames[imageId])
                        savedCounter += 1
                    except exception as e:
                        self.logger.error(f"Failed to save image number: {idx} because {e} occured.")
                        continue
            
            self.logger.info(f"Saved {savedCounter} images for transformation: {transformation}.")


    def transformToGrayscale(self, idx):
        '''
        Transforms an image to grayscale given an ID.

            Parameters:
                idx (int): Image ID

            Returns:
                grayImage (numpy array): Grayscale image
        '''
        image, _, _, _, _ = self.dataset[idx]
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
        image, _, _, _, _ = self.dataset[idx]
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invertedGrayImage = 255.0 - grayImage
        invertedGrayImage = np.float32(invertedGrayImage)

        return cv2.cvtColor(invertedGrayImage, cv2.COLOR_GRAY2BGR)
    
    
    def disk(self, radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
    
    
    def transformToshot_noise(self, idx, c=10):
        x, _, _, _, _ = self.dataset[idx]
        x = x / 255.
        print(x.shape)
        new_img= np.clip(x+(np.random.poisson( size=x.shape, lam=.1)), 0, 1) * 255
        new_img=np.float32(new_img)
        return cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    
    def transformTogaussian_noise(self, idx, severity=1):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.array(x) / 255.
        new_img= np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
        new_img=np.float32(new_img)
        return cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    
    def transformToimpulse_noise(self, idx, severity=4):
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return cv2.cvtColor(np.float32(np.clip(x, 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    def transformTospeckle_noise(self, idx, severity=1):
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.array(x) / 255.
        return cv2.cvtColor(np.float32(np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    
    def transformTodefocus_blur(self, idx, severity=1):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.array(x) / 255.
        kernel = self.disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        return cv2.cvtColor(  np.float32(np.clip(channels, 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    def transformToglass_blur(self, idx, severity=1):
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(x.shape[0] - c[1], c[1], -1):
                for w in range(x.shape[1] - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

        return cv2.cvtColor(np.float32(np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    def clipped_zoom(img, zoom_factor):
        h = img.shape[0]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / float(zoom_factor)))

        w= img.shape[1]
        ch2 = int(np.ceil(w / float(zoom_factor)))
        top = (h - ch) // 2
        side= (w - ch2) // 2
        img = scizoom(img[top:top + ch, side:side + ch2], (zoom_factor, zoom_factor, 1), order=1)
        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2
        trim_side = (img.shape[1] - w) // 2


        return img[trim_top:trim_top + h, trim_side:trim_side + w]
    
    
    def transformTozoom_blur(self, idx, severity=1):
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        #print(out.shape)
        for zoom_factor in c:
            temp = self.clipped_zoom(x, zoom_factor)
            #print(temp.shape)
            out += temp


        x = (x + out) / (len(c) + 1)
        #return np.clip(x, 0, 1) * 255
        return cv2.cvtColor(np.float32(np.clip(x, 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    
    
    # class MotionImage(WandImage):
    #     def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
    #         wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
            
    def transformTosnow(self, idx, severity=1):
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.array(x, dtype=np.float32) / 255.
        snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

        snow_layer = self.clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.
        snow_layer = snow_layer[..., np.newaxis]

        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
        return cv2.cvtColor(np.float32(np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    
    def transformTobrightness(self, idx, severity=5):
        c = [.1, .2, .3, .4, .5][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)
        return cv2.cvtColor(np.float32(np.clip(x, 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    def transformTocontrast(self, idx, severity=1):
        c = [0.4, .3, .2, .1, .05][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        #return np.clip((x - means) * c + means, 0, 1) * 255
        return cv2.cvtColor(np.float32(np.clip((x - means) * c + means, 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    def transformToelastic_transform(self, idx, severity=5):
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
        image, _, _, _, _ = self.dataset[idx]

        image = np.array(image, dtype=np.float32) / 255.
        shape = image.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        #return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255
        return cv2.cvtColor(np.float32(np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255), cv2.COLOR_BGR2RGB)
    
    
    def transformTopixelate(self, idx, severity=5):
        c = [0.6, 0.5, 0.4, 0.3, 0.15][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        width = int(x.shape[1] * c)
        height = int(x.shape[0] * c)
        dim = (width, height)
        resized = cv2.resize(x, dim, interpolation = cv2.INTER_AREA)
        return cv2.cvtColor(np.float32(resized), cv2.COLOR_BGR2RGB)
    
    def transformTojpeg_compression(self, idx, severity=5):
        c = [25, 18, 15, 10, 7][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        cv2.imwrite("parrot_saved.jpg", x, [int(cv2.IMWRITE_JPEG_QUALITY), c]) 
        temp = imread("parrot_saved.jpg")
        return temp
    
    def transformTospatter(self, idx, severity=4):
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
             (0.65, 0.3, 3, 0.68, 0.6, 0),
             (0.65, 0.3, 2, 0.68, 0.5, 0),
             (0.65, 0.3, 1, 0.65, 1.5, 1),
             (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.array(x, dtype=np.float32) / 255.

        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                    42 / 255. * np.ones_like(x[..., :1]),
                                    20 / 255. * np.ones_like(x[..., :1])), axis=2)

            color *= m[..., np.newaxis]
            x *= (1 - m[..., np.newaxis])

            return cv2.cvtColor(np.float32(np.clip(x + color, 0, 1) * 255), cv2.COLOR_BGR2RGB)

    
    def transformTosaturate(self, idx, severity=1):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
        x, _, _, _, _ = self.dataset[idx]
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)  
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)
        return cv2.cvtColor(np.float32(np.clip(x, 0, 1) * 255), cv2.COLOR_BGR2RGB)



# wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
#                                               ctypes.c_double,  # radius
#                                               ctypes.c_double,  # sigma
#                                               ctypes.c_double)  # angle


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
