import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

# This file was removed before pushing to github.
RAW_DATA_FILE = os.path.join(DATA_DIR, "finefoods.txt")

TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train.txt")
VALID_DATA_FILE = os.path.join(DATA_DIR, "valid.txt")

# This file was removed before pushing to github.
TEST_DATA_FILE = os.path.join(DATA_DIR, "test.txt")

# The column to make predictions for.
PREDICTION_COLUMN = "score"

RESULTS_FILE = os.path.join(BASE_DIR, "results.json")
