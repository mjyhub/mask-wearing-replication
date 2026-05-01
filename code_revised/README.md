# Revised Code Workflow

Run the scripts in order:

```bash
python3 02_load_and_check_data.py
python3 03_cleaning_pipeline.py
python3 04_feature_engineering.py
python3 05_policy_merge_check.py
python3 06_train_valid_split.py
```

Optional baseline model training:

```bash
python3 07_model_training_cv.py
```

The scripts read raw files from `../raw_data/` and write the same style of
outputs as the original replication workflow into `../data/`, `../results/`,
and `../models/`.

Expected data audit:

```text
raw data: 53,833 x 513
cleaned data: 40,136 x 26
preprocessed data: 40,136 x 65
before mandate: 14,945
after mandate: 25,191
train: 32,108
test: 8,028
```

