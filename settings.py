import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

# This file was removed before pushing to github.
RAW_DATA_FILE = os.path.join(DATA_DIR, "clean_cars.csv")

TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train.csv")
VALID_DATA_FILE = os.path.join(DATA_DIR, "valid.csv")

# This file was removed before pushing to github.
TEST_DATA_FILE = os.path.join(DATA_DIR, "test.csv")

# The column to make predictions for.
PREDICTION_COLUMN = "GHG"

RESULTS_FILE = os.path.join(BASE_DIR, "results.json")
