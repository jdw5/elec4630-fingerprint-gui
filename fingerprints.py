from os import path
import utils
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *
from ipywidgets import interact

def read(path: str):
    fingerprint = cv.imread('samples/sample_1_1.png', cv.IMREAD_GRAYSCALE)
    # img = show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}')
    cv.imshow("Fingerprint", fingerprint)
    cv.waitKey(0) 
    cv.destroyAllWindows() 

if __name__ == "__main__":
    read('hi')