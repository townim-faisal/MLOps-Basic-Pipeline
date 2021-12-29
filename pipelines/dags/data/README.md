# Data Pipeline

Raw data will be stored in `storage\data-lake`. Preprocessed data will be save in `storage\data-warehouse`. Every preprocessed data has a version name defined in [param.yaml](params/param.yaml)'s `version` key.


`eda.py` -> Exploratory data anaklysis
`preprocess.py` -> Preprocessing data warehouse's data
`validate.py` -> Validating preprocessed data


