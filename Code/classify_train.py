import os
from time import time
import data_preprocess as dv
import pandas as pd
#import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

class Solution(object):
    def __init__(self):

        self.X_train,self.X_test,self.y_train,self.y_test,self.dataframe_all=dv.preprocess(os.path.join(os.getcwd(),"shuffled-full-set-hashed.csv"))

    def visualize(self):
        dv.visualization(self.X_test,self.y_test)

    def train_classifier(self,clf):
        ''' Fits a classifier to the training data. '''

        # Start the clock, train the classifier, then stop the clock
        start = time()
        clf.fit(self.X_train, self.y_train)
        end = time()

        # Print the results
        print("Trained model in {:.4f} seconds".format(end - start))
        return clf

    def predict_labels(clf, features, target):
        ''' Makes predictions using a fit classifier based on F1 score. '''

        # Start the clock, make predictions, then stop the clock
        start = time()
        y_pred = clf.predict(features)

        end = time()
        # Print and return results
        print("Made predictions in {:.4f} seconds.".format(end - start))

        return f1_score(target, y_pred, labels=range(14), average='micro'), sum(target == y_pred) / float(len(y_pred))

    def train_predict(self,clf):
        ''' Train and predict using a classifer based on F1 score. '''

        # Indicate the classifier and the training set size
        #print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(self.X_train)))

        # Train the classifier
        self.train_classifier(clf)

        # Print the results of prediction for both training and testing
        f1, acc = self.predict_labels(clf, self.X_train, self.y_train)
        print(f1, acc)
        print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

        f1, acc = self.predict_labels(clf, self.X_test, self.y_test)
        print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))

    # def grid_search(self):
    #
    #
    #     # parameters list we wish to tune
    #     parameters = {'learning_rate': [0.1],
    #                   'n_estimators': [40],
    #                   'max_depth': [3],
    #                   'min_child_weight': [3],
    #                   'gamma': [0.4],
    #                   'subsample': [0.8],
    #                   'colsample_bytree': [0.8],
    #                   'scale_pos_weight': [1],
    #                   'reg_alpha': [1e-5]
    #                   }
    #
    #     # Initialize the classifier
    #     clf = xgb.XGBClassifier(seed=82)
    #
    #     # Make an f1 scoring function using 'make_scorer'
    #     f1_scorer = make_scorer(f1_score, pos_label=1)
    #
    #     # Perform grid search on the classifier using the f1_scorer as the scoring method
    #     grid_obj = GridSearchCV(clf,
    #                             scoring=f1_scorer,
    #                             param_grid=parameters,
    #                             cv=5)
    #
    #     # Fit the grid search object to the training data and find the optimal parameters
    #     grid_obj = grid_obj.fit(self.X_train, self.y_train)
    #
    #     # Get the estimator
    #     clf = grid_obj.best_estimator_
    #
    #     return clf



if __name__=="__main__":
    s=Solution()
    s.train_predict(s.train_classifier(SVC(random_state = 912, kernel='rbf')))

