# Data Pipeline

Raw data will be stored in `storage\data-lake`. Preprocessed data will be save in `storage\data-warehouse`. Every preprocessed data has a version name defined in [param.yaml](params/param.yaml)'s `version` key.


`eda.py` -> Exploratory data anaklysis
`preprocess.py` -> Preprocessing data warehouse's data
`validate.py` -> Validating preprocessed data

# Params
A sample param file is [here](params/param.yaml).
```
lake_dir: <absolute path for data lake>
warehouse_dir: <absolute path for data warehouse>
version: <data version name>
log_folder: <absolute path of log folder>
figure_folder: <absolute path of figure folder>

extention: <image extention>
num_classes: <number of class need to be in dataset>
class_names: <array of class names>
seed: <seed number>

# eda
blur_threshold: 0.95
similarity_threshold: 1.05

# preprocess
test_ratio: <ratio of test set in dataset>
delete_blur_image: <boolean>
delete_duplicate_image: <boolean>
```



