from skimage.metrics import structural_similarity as compare_ssim
from collections import Counter
import numpy as np
from PIL import Image
import ffmpeg, shutil, glob
import multiprocessing as mp
import pandas as pd
import re
import os, sys
import yaml
import cv2
import matplotlib.pyplot as plt
from validate import parallel_laplacian_variance


# parameter
with open("params/param.yaml", "r") as config_file:
    params = yaml.safe_load(config_file)
print("Parameters:", params)


# how much blur image
files = [file for file in glob.glob(f"{params['lake_dir']}/*.jpg")]
print(files)
pool = mp.Pool(mp.cpu_count())  
blurriness = [pool.apply(parallel_laplacian_variance, args=(file,)) for file in files]
pool.close() 

median_blur = float(np.median(blurriness))
min_blur = float(np.min(blurriness))
max_blur = float(np.max(blurriness))
print("Median Blur (Laplacian Variance): " + str(median_blur))
blur_cutoff = median_blur*0.95 #+ ((1-average_blur)*0.1)
print("Blur Cutoff (Laplacian Variance): " + str(blur_cutoff))


# how much duplicate image


# data distribution per class


