Unite!
----------------

This is an experimental machine learning contest.  Instead of having everyone make an algorithm, and then picking the one with the lowest error as the "winner", we're going to collaboratively design an approach.  There's a focus on execution time and code readability (enforced through pep8), not just error rate.  Points are assigned based on a few criteria, including opening a pull request, having a pull request merged, and having a lower error rate than anyone else.  You can find the point leaderboard [here](http://unite.dataquest.io/f/leaderboard/dataquestio/unite).

## The problem

We'll be predicting the score (0-5) given to a product by an amazon reviewer based on the text and summary of the review.  The data is originally from the Stanford Large Network Dataset Collection.  

We'll be using root mean squared error as the evaluation metric.  The algorithm performance will be evaluated on a hidden test set.

## Contributing

You can contribute by forking this repo, making some changes, and then opening a pull request.  We have a server that will automatically run and check your code, and assign you points if it passes.  You can also run the tests yourself with `test.py`.  All kinds of changes are welcome, including adding comments, refactoring code, improving algorithms, and anything else.  The idea is to allow anyone to contribute, regardless of machine learning experience.

How to get started
-------------------

Getting started is a pretty simple process.  Before you get started, you should have python 2.7 installed.  Using a virtualenv is optional but recommended.

1. Fork this repo to your github account.
2. Clone your fork locally (`git clone git@github.com:YOUR_USERNAME/unite.git`)
3. Get into the unite folder (`cd unite`)
4. Install the python requirements (`pip install -r requirements.txt`)
5. Run the tests (`python test.py data/train.txt data/valid.txt`)
    * This will train the current algorithm using the training set, and then make predictions on the validation set.
6. Add your contributions!  Make the code better.
7. When you're ready, and the tests pass, make a pull request from your fork to the main repo.
    * If your code takes more than 15 minutes to run, it will result in an error when the server runs it.
    * The maximum available memory for your code is 4 gigs.

Point Assignment
------------------

Right now, you get:

* 2 points for opening a pull request.
* 10 points if your error is lower than the global minimum (across all open, unmerged, and merged pull request)
    * Note: You have to open a pull request to get this -- just working on a fork isn't enough.
* 4 points for getting a pull request merged.

File structure
-------------------

* `data` -- this folder contains the raw training and validation data.
    * `train.txt` -- raw training data.
    * `valid.txt` -- raw validation data. 
* `algo.py` -- this contains the main algorithm class that is used to train and evaluate.
    * The algorithm doesn't have to be restricted to this file -- feel free to make other files and import from them.
* `settings.py` -- this contains various settings that can be called from other modules.
* `split_data.py` -- this file documents how the data was originally split up and anonymized.
    * Don't edit this file.
* `test.py` -- this file allows you to test the error rate of your approach, and is used to evaluate your submission.
    * Don't edit this file -- it's used to compute error.

Merge conflicts
------------------

We'll be merging pull requests on a "first reasonable pull request will be merged" basis.  This will probably lead to cases where later PRs can't be merged due to conflicts.  If this happens, you can close your PR, update it, and then open a new one.

If you want to work on a unique approach on your own unmergeable fork, you can feel free to do that and periodically open pull requests -- this will enable you to get the 10 points, but will prevent you from getting any merge points.

Suggestions?
-------------------

This is an experiment and will likely evolve a lot over time.  If you have any questions or suggestions, please open an issue.  You can also think of the issues as a discussion board to talk about new approaches.