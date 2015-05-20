"""
This file exists to show how the data set was split into training, validation,
and test sets.
"""

import math
import random
import settings


_EXCLUDE_FIELDS = (
    "review/profileName",
    "review/userId",
    "review/time",
)


def cleanup_review(data):
    """Remove a few fields from the reviews."""
    lines = data.split("\n")
    return "\n".join(line for line in lines
                     if not any(line.startswith(field)
                                for field in _EXCLUDE_FIELDS))


if __name__ == "__main__":
    # Open the raw data file that contains all the rows.
    with open(settings.RAW_DATA_FILE) as f:
        DATA = f.read()

    # Split the data into reviews and randomly downsample.
    REVIEWS = random.sample([cleanup_review(review)
                             for review in DATA.split("\n\n")], 40000)

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
