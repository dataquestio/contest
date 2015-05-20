"""
This contains the main algorithm class that is used to train and evaluate. The
algorithm doesn't have to be restricted to this file -- feel free to make other
files and import from them.
"""

from __future__ import division

import re
from pandas import DataFrame
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


class Algorithm(object):
    """
    This class is imported and called by test.py.  It is used to train the
    classifier and run predictions.  You can edit the class (and add and import
    other files in the repo, also).  Be sure not to change the function
    signature of __init__, train, predict, generate_df, or generate_features.
    If you do, your submission will fail.
    """

    def __init__(self):
        """
        Initialize any specific algorithm-related variables here.
        """

        # Create a new classifier.
        self.clf = Ridge()
        # Create a vectorizer to extract features.
        # Important to make this a class attribute, as the vocab needs to be
        # the same for train and prediction sets.
        self.vectorizer = Pipeline([
            ('vect', CountVectorizer(min_df=20, stop_words="english",
                                     ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer())
        ])
        self.collist = []

    @classmethod
    def generate_df(cls, data):
        """
        Generate a pandas dataframe from the raw input data.
        :param data: Review data read in from text files.
        :return: A pandas dataframe, where each row is a review, and contains
          all the needed information.
        """
        # Split the data into sections (what will become data_frame rows)
        rows = data.split("\n\n")

        matrix = []
        # Read in the data section by section.
        for row in rows:
            new_row = []
            # Split each section into lines.
            for i, line in enumerate(row.strip().split("\n")):
                # Strip out the beginning of the line, which indicates which
                # field it is.
                line = re.sub("[a-z]+/[a-zA-Z]+: ", "", line)
                # Convert to unicode.
                line = unicode(line, errors='replace')

                # Some scores are not numbers -- convert them.
                if i == 2:
                    if line == "1/2":
                        line = "2"
                    elif line == "0/0":
                        line = "0"
                    elif line == "2/2":
                        line = "5"
                new_row.append(line)
            # Add the row into the new matrix.
            matrix.append(new_row[:5])
        # Convert the list of lists into a pandas dataframe
        columns = ["product_id", "helpfulness", "score", "summary", "text"]
        data_frame = DataFrame(matrix, columns=columns)
        data_frame["score"] = data_frame["score"].astype(float)
        data_frame["text"] = [t.lower() for t in data_frame["text"]]
        return data_frame

    def generate_features(self, data_frame, stage="train"):
        """
        Generate the textual features needed to train or predict data.
        Called when generating the dataframe to either train the algorithm, or
        predict using it.
        :param data_frame: A dataframe containing the extracting test or
          training data.
        :param stage: A string indicating what type of features should be made.
          Either "train" for training set, or "test" for test set.
        :return: A matrix or pandas dataframe containing the features.
        """
        # Extract features.
        text = data_frame["text"]
        if stage == "train":
            algorithmic_features = self.vectorizer.fit_transform(text)
        else:
            algorithmic_features = self.vectorizer.transform(text)

        def count_words_negated(text, words_to_check):
            negation_words = ["not", "don't", "didn't", "didnt", "wasnt", "wasn't"]
            negation_words_regex = "|".join(negation_words)
            words_to_check_regex = "|".join(words_to_check)
            text_sentences = re.split("[?.!]", text) #simplifies checking words are in same sent
            my_regex = r"\b(%s)\b.*\b(%s)\b|\b(%s)\b.*\b(%s)\b"%(negation_words_regex, words_to_check_regex, \
                                                words_to_check, negation_words_regex)
            out = len(re.findall(my_regex, text))
            return(out)

        # Define some functions that can transform the text into features.
        good_words = ["good", "great", "better", "best", "efficient", "sweet",
                        "delicious", "like", "love", "thanks", "perfect"]
        bad_words = ["bad", "worse"]

        transform_functions = [
            ("length", len),
            ("exclams", lambda x: x.count("!")),
            ("question_marks", lambda x: x.count("?")),
            ("sentences", lambda x: x.count(".")),
            # Add one as a smooth.
            ("words_per_sentence",
             lambda x: x.count(" ") / (x.count(".") + 1)),
            ("letters_per_word", lambda x: len(x) / (x.count(" ") + 1)),
            ("commas", lambda x: x.count(",")),
            ("negated_good_words", lambda x: count_words_negated(x, good_words)),
            ("negated_bad_words", lambda x: count_words_negated(x, bad_words))
        ]
        hand_chosen_features = DataFrame()

        for col in ["text", "summary"]:
            for name, func in transform_functions:
                transformed = data_frame[col].apply(func)
                hand_chosen_features["{0}_{1}".format(col, name)] = transformed

        helpful_yes = data_frame.helpfulness.apply(lambda x: x.split("/")[0])
        hand_chosen_features['helpful_yes'] = helpful_yes.astype('int')

        helpful_total = data_frame.helpfulness.apply(lambda x: x.split("/")[1])
        hand_chosen_features['helpful_total'] = helpful_total.astype('int')

        features = hstack([algorithmic_features, hand_chosen_features])

        if stage == "train":
            # Select 2000 "best" columns based on chi squared.
            selector = SelectKBest(chi2, k=2000)
            selector.fit(features, data_frame["score"])
            self.collist = selector.get_support().nonzero()

        # Grab chi squared selected column subset.
        features = features.tocsc()[:, self.collist[0]].todense()

        return features

    def train(self, feats, to_predict):
        """
        Train a prediction algorithm.
        :param feats: The training features.
        :param to_predict: A pandas series or numpy array containing the column
          to predict.
        :return: None
        """

        # Fit the regressor (train it) with our selected columns, and try to
        # predict the to_predict column.
        self.clf.fit(feats, to_predict)

    def predict(self, feats):
        """
        Predict values for new data.
        :param feats: The features to make predictions using.  Must have the
          exact same columns as the feats passed to the train function.
        :return: The predictions for the to_predict column on data_frame.
        """

        # Generate the predictions.
        # (false positive) pylint: disable=E1101
        predictions = self.clf.predict(feats).clip(0.1, 4.9)
        return predictions
