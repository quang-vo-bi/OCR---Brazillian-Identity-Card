import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
import helper_function

class Preprocessor:
    def __init__(self, params):
        self.params = params

    def transform(self, imgArr):
        # Convert to Grey Scale and Gaussian Blurring
        if self.params['gray_scale']:
            imgArr = cv2.cvtColor(imgArr, cv2.COLOR_BGR2GRAY)
        if self.params['gaussian_blurring']:
            imgArr = cv2.GaussianBlur(imgArr, ksize=self.params['ksize'], sigmaX=0, sigmaY=0)

        # Equalization and Thresholding
        if self.params['equalization']:
            clahe = cv2.createCLAHE(clipLimit=self.params['clip_limit'], tileGridSize=self.params['tile_grid_size'])
            imgArr = clahe.apply(imgArr)
        if self.params['thresholding']:
            _, imgArr = cv2.threshold(imgArr, thresh=self.params['thresh'], maxval=self.params['maxval'], type=self.params['type_'])

        return imgArr





# def correct_orient(imgArr):
#     rgb = cv2.cvtColor(imgArr, cv2.COLOR_BGR2RGB)
#     results = pytesseract.image_to_osd(rgb, output_type=Output.DICT, config='— psm 0')
#     rotated = helper_function.rotate_image(imgArr, angle=results["rotate"])
#     return rotated


