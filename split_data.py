"""
This file exists to show how the data set was split into training, validation, and test sets.
"""

import pandas as pd
import settings
import math
import random


if __name__ == "__main__":
    # Open the raw data file that contains all the rows.
    with open(settings.RAW_DATA_FILE) as f:
        data = f.read()

    # Split the data into reviews.
    reviews = data.split("\n\n")
    # Randomly shuffle the reviews.
    random.shuffle(reviews)

    # Downsample the data to run faster
    reviews = reviews[:40000]
    # Remove a few fields from the reviews.
    for i, r in enumerate(reviews):
        r = r.split("\n")
        reviews[i] = "\n".join([l for l in r if not l.startswith("review/profileName") and not l.startswith("review/userId") and not l.startswith("review/time")])

    # Set a subset size -- valid set and test set are each 25% of the data, train set is 50%.
    subset_size = int(math.floor(len(reviews) / 4))

    # Split the data up into sets.
    valid_set = reviews[:subset_size]
    test_set = reviews[subset_size:(subset_size*2)]
    train_set = reviews[(subset_size*2):]

    # Write all of the sets to their respective files.
    with open(settings.TRAIN_DATA_FILE, "w+") as f:
        f.write("\n\n".join(train_set))

    with open(settings.VALID_DATA_FILE, "w+") as f:
        f.write("\n\n".join(valid_set))

    with open(settings.TEST_DATA_FILE, "w+") as f:
        f.write("\n\n".join(test_set))
