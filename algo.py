from sklearn.linear_model import LogisticRegression


class Algorithm(object):
    """
    This class is imported and called by test.py.
    It is used to train the classifier and run predictions.
    You can edit the class (and add and import other files in the repo, also).
    Be sure not to change the function signature of __init__, train, or predict.
    If you do, your submission will fail.
    """

    def __init__(self):
        """
        Initialize any specific algorithm-related variables here.
        """

        # Create a new classifier.
        self.clf = LogisticRegression()
        self.collist = []

    def train(self, df, to_predict):
        """
        Train a prediction algorithm.
        :param df: A pandas dataframe containing the training data.
        :param to_predict: A string containing the name of the column that contains the variable to predict.
        :return: None
        """

        # Make a list of columns to use for training.
        self.collist = df.columns.tolist()
        # Remove the to_predict column from our list.
        self.collist.remove(to_predict)

        # Remove any columns we don't want to use in training.
        non_predictors = ["Car"]
        for col in non_predictors:
            self.collist.remove(col)

        # Fit the classifier (train it) with our selected columns, and try to predict the to_predict column.
        self.clf.fit(df[self.collist], df[to_predict])

    def predict(self, df):
        """
        Predict values for new data.
        :param df: A pandas dataframe containing at least all of the columns in self.collist.
        :return: The predictions for the to_predict column on df.
        """

        # Generate the predictions.
        predictions = self.clf.predict(df[self.collist])
        return predictions
