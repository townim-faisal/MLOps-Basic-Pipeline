import numpy as np
import glob
import multiprocessing as mp
import pandas as pd
import os, sys
import yaml
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from validate import parallel_laplacian_variance, parallel_compare_images


# parameter
with open("params/param.yaml", "r") as config_file:
    params = yaml.safe_load(config_file)
print("Parameters:", params)

# log file
log = {}

# list of files
files = sorted([file for file in glob.glob(f"{params['lake_dir']}/*{params['extention']}")])
print("Total files:", len(files))

# save files len in log
log['files'] = {
    'dir': params['lake_dir'],
    'count': len(files)
}

# how much blur image
blurriness = []
for i, file in tqdm(enumerate(files), total=len(files), desc="Blur: "):
    blurriness.append(parallel_laplacian_variance(file))
median_blur = float(np.median(blurriness))
min_blur = float(np.min(blurriness))
max_blur = float(np.max(blurriness))
print("Median Blur (Laplacian Variance): " + str(median_blur))
blur_cutoff = median_blur*params['blur_threshold'] #+ ((1-average_blur)*0.1)
print("Blur Cutoff (Laplacian Variance): " + str(blur_cutoff))

# save blur infoin log
log['blur'] = {
    'median_blur': median_blur,
    'min_blur': min_blur,
    'max_blur': max_blur,
    'blur_cutoff': blur_cutoff
}


# how much duplicate image
diff = []
for i in tqdm(range(len(files)-1), total=len(files)-1, desc="Duplicate: "):
    diff.append(parallel_compare_images(i, files))
median_diff = float(np.median(diff))
print('Median Similarity Cutoff (OpenCV Compare Images):', median_diff)
diff_cutoff = median_diff*params['similarity_threshold']
print('Similarity Cutoff (OpenCV Compare Images):', diff_cutoff)

# save duplicate info in log
log['duplicate'] = {
    'median_diff': median_diff,
    'diff_cutoff': diff_cutoff
}

# data distribution per class
num_classes = params['num_classes']
class_names = {name: 0 for name in params['class_names']}

for i, file in tqdm(enumerate(files), total=len(files), desc="Distribution: "):
    for name in class_names:
        if name in file:
            class_names[name] = class_names[name]+1
print("Class distribution:", class_names)

# save class distribution in log
log['class_distribution'] = {
    'class_names': params['class_names'],
    'dist': class_names
}

print('Log:', log)

with open(os.path.join(params['log_folder'], f"eda_log_{params['version']}.json"), 'w', encoding='utf-8') as f:
    json.dump(log, f, ensure_ascii=False, indent=4)
