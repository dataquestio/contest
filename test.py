"""
Run tests on the data.  Used when submitting the answer.

Usage -- python test.py TRAINING_FILE_PATH PREDICTION_FILE_PATH
"""
from __future__ import print_function

import StringIO
import argparse
import json
import math
import sys
import time

from pep8 import StyleGuide
from sklearn.metrics import mean_squared_error

import algo
import settings

# Parse input arguments.
PARSER = argparse.ArgumentParser(description='Test code to see if it works.')
PARSER.add_argument('train_file', help='The training file to use.')
PARSER.add_argument('prediction_file', help='The file to make predictions on.')
PARSER.add_argument('--write', action="store_true",
                    help='Whether to write results to a file.')

if __name__ == "__main__":
    ARGS = PARSER.parse_args()

    # Read the training file.
    with open(ARGS.train_file) as f:
        TRAIN_DATA = f.read()

    with open(ARGS.prediction_file) as f:
        PREDICTION_DATA = f.read()

    START_TIME = time.time()
    # Initialize the algorithm class.
    ALG = algo.Algorithm()

    # Generate a dataframe from the train text.
    TRAIN_DF = ALG.generate_df(TRAIN_DATA)
    # Get the features from the dataframe
    TRAIN_FEATURES = ALG.generate_features(TRAIN_DF, stage="train")
    # Train the algorithm using the training features.
    ALG.train(TRAIN_FEATURES, TRAIN_DF["score"])

    # Generate a prediction dataframe.
    PREDICTION_DF = ALG.generate_df(PREDICTION_DATA)
    # Generate features from the dataframe
    PREDICTION_FEATURES = ALG.generate_features(PREDICTION_DF, stage="test")
    # Make predictions using the prediction dataframe.
    PREDICTIONS = ALG.predict(PREDICTION_FEATURES)

    # Find how long it took to execute.
    EXECUTION_TIME = time.time() - START_TIME
    print("Execution time was {0} seconds.\n".format(EXECUTION_TIME))

    # We're using RMSE as a metric.
    ACTUAL_VALUES = PREDICTION_DF[settings.PREDICTION_COLUMN]
    ERROR = math.sqrt(mean_squared_error(PREDICTIONS, ACTUAL_VALUES))
    print("Found root mean squared error of: {0}\n".format(ERROR))

    # Setup a buffer to capture pep8 output.
    BUFFER = StringIO.StringIO()
    sys.stdout = BUFFER

    # Initialize and run a pep8 style checker.
    PEP8STYLE = StyleGuide(ignore="E121,E123,E126,E226,E24,E704,E501")
    PEP8STYLE.input_dir(settings.BASE_DIR)
    REPORT = PEP8STYLE.check_files()

    # Change stdout back to the original version.
    sys.stdout = sys.__stdout__

    PEP8_RESULTS = BUFFER.getvalue()
    if REPORT.total_errors > 0:
        print("Pep8 violations found!  They are shown below.")
        print("----------------------")
        print(PEP8_RESULTS)

    # Write all the results to a file if needed.
    if ARGS.write:
        WRITE_DATA = {
            "error": ERROR,
            "execution_time": EXECUTION_TIME,
            "pep8_results": PEP8_RESULTS
        }
        with open(settings.RESULTS_FILE, "w+") as f:
            json.dump(WRITE_DATA, f)
