"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

### We are no longer use abalone dataset, but rather the league of legends playoffs dataset. 
### We are performing unsupervised learning to group the players together
### https://www.kaggle.com/datasets/smvjkk/league-of-legends-lec-spring-playoffs-2024-stats?resource=download



## Our new dataset has headers

# Since we get a headerless CSV file we specify the column names here.
# feature_columns_names = [
#     "sex",
#     "length",
#     "diameter",
#     "height",
#     "whole_weight",
#     "shucked_weight",
#     "viscera_weight",
#     "shell_weight",
# ]
# label_column = "rings"

import numpy as np

feature_columns_dtype = {
    "Player": str,
    "Role": str,
    "Team": str,
    "Opponent_Team": str,
    "Opponent_Player": str,
    "Date": str,
    "Round": int,
    "Day": int,
    "Patch": str,
    "Stage": str,
    "No_Game": int,
    "all_Games": int,
    "Format": int,
    "Game_of_day": int,
    "Side": str,
    "Time": str,
    "Ban": str,
    "Ban_Opponent": str,
    "Pick": str,
    "Pick_Opponent": str,
    "Champion": str,
    "Champion_Opponent": str,
    "Outcome": str,
    "Kills_Team": int,
    "Turrets_Team": int,
    "Dragon_Team": int,
    "Baron_Team": int,
    "Level": int,
    "Kills": int,
    "Deaths": int,
    "Assists": int,
    "KDA": np.float64,
    "CS": int,
    "CS in Team's Jungle": int,
    "CS in Enemy Jungle": int,
    "CSM": np.float64,
    "Golds": int,
    "GPM": np.float64,
    "GOLD%": np.float64,
    "Vision Score": int,
    "Wards placed": int,
    "Wards destroyed": int,
    "Control Wards Purchased": int,
    "Detector Wards Placed": int,
    "VSPM": np.float64,
    "WPM": np.float64,
    "VWPM": np.float64,
    "WCPM": np.float64,
    "VS%": np.float64,
    "Total damage to Champion": int,
    "Physical Damage": int,
    "Magic Damage": int,
    "True Damage": int,
    "DPM": np.float64,
    "DMG%": int,
    "K+A Per Minute": np.float64,
    "KP%": int,
    "Solo kills": int,
    "Double kills": int,
    "Triple kills": int,
    "Quadra kills": int,
    "Penta kills": int,
    "GD@15": int,
    "CSD@15": int,
    "XPD@15": int,
    "LVLD@15": int,
    "Objectives Stolen": int,
    "Damage dealt to turrets": int,
    "Damage dealt to buildings": int,
    "Total heal": int,
    "Total Heals On Teammates": int,
    "Damage self mitigated": int,
    "Total Damage Shielded On Teammates": int,
    "Time ccing others": int,
    "Total Time CC Dealt": int,
    "Total damage taken": int,
    "Total Time Spent Dead": int,
    "Consumables purchased": int,
    "Items Purchased": int,
    "Shutdown bounty collected": int,
    "Shutdown bounty lost": int
}


## Removing labels since we are doing unsupervised learning

# label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/abalone-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=None, # FORME: What does header do?
        dtype=feature_columns_dtype,
    )
    os.unlink(fn)

    logger.debug("Defining transformers.")
    numeric_features = [col for col, dtype in feature_columns_dtype.items() if dtype in [int, float, np.float64]]
    categorical_features = [col for col, dtype in feature_columns_dtype.items() if dtype == str]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Applying transforms.")
    y = df.pop("rings")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
