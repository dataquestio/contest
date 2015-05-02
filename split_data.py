"""
This file exists to show how the data set was split into training, validation,
and test sets.
"""

import math
import random
import settings


if __name__ == "__main__":
    # Open the raw data file that contains all the rows.
    with open(settings.RAW_DATA_FILE) as f:
        DATA = f.read()

    # Split the data into reviews.
    REVIEWS = DATA.split("\n\n")
    # Randomly shuffle the reviews.
    random.shuffle(REVIEWS)

    # Downsample the data to run faster
    REVIEWS = REVIEWS[:40000]
    # Remove a few fields from the reviews.
    for i, review in enumerate(REVIEWS):
        lines = review.split("\n")
        REVIEWS[i] = "\n".join([line for line in lines
                                if not line.startswith("review/profileName")
                                and not line.startswith("review/userId")
                                and not line.startswith("review/time")])

    # Set a subset size
    # valid set and test set are each 25% of the data, train set is 50%.
    SUBSET_SIZE = int(math.floor(len(REVIEWS) / 4))

    # Split the data up into sets.
    VALID_SET = REVIEWS[:SUBSET_SIZE]
    TEST_SET = REVIEWS[SUBSET_SIZE:(SUBSET_SIZE * 2)]
    TRAIN_SET = REVIEWS[(SUBSET_SIZE * 2):]

    # Write all of the sets to their respective files.
    with open(settings.TRAIN_DATA_FILE, "w+") as f:
        f.write("\n\n".join(TRAIN_SET))

    with open(settings.VALID_DATA_FILE, "w+") as f:
        f.write("\n\n".join(VALID_SET))

    with open(settings.TEST_DATA_FILE, "w+") as f:
        f.write("\n\n".join(TEST_SET))
