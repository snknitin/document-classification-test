import pandas as pd
import numpy as np
import shared_utils as su
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit



def preprocess(filename):
    """

    :return: Test and train splits after processing
    """
    # step 1: Load the data and include column headers
    dataframe_all = pd.read_csv(filename, sep=",")
    dataframe_all.columns = ["document_label", "word_values"]
    dataframe_all = dataframe_all.dropna(subset=['word_values'])

    return dataframe_all
    # step 2 Make a stratified sample of the classes and split into test and train sets

def create_train_test(dataframe_all):
    """
    Create train test split from the incoming data frame.
    During training this will create both sets, during production we only need test
    :param dataframe_all:
    :return:
    """
    label_encoder=LabelEncoder()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in split.split(dataframe_all['word_values'], dataframe_all['document_label']):
        strat_train_set = dataframe_all.loc[train_index]
        strat_test_set = dataframe_all.loc[test_index]

    strat_train_set = strat_train_set.dropna(subset=['word_values'])
    strat_test_set = strat_test_set.dropna(subset=['word_values'])
    pipe=su.pipe()
    x_train, y_train = pipe.fit_transform(strat_train_set), label_encoder.fit_transform(
        strat_train_set['document_label'])
    x_test, y_test = pipe.transform(strat_test_set), label_encoder.fit_transform(
        strat_test_set['document_label'])

    return x_train,x_test,y_train,y_test

def prod_test(dataframe):
    """
    Given the documents for test time, we need to predict the correct class.
    So the features need to be processed to predict wih the loaded model
    :param dataframe:
    :return:
    """
    dataframe.columns = ["word_values"]
    dataframe = dataframe.dropna(subset=['word_values'])
    return  su.pipe().transform(dataframe)

