import pandas as pd
import numpy as np
import shared_utils as su
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
label_encoder=LabelEncoder()



def preprocess(filename):
    """

    :return: Test and train splits after processing
    """
    # step 1: Load the data and include column headers
    dataframe_all = pd.read_csv(filename, sep=",")
    dataframe_all.columns = ["document_label", "word_values"]
    dataframe_all = dataframe_all.dropna(subset=['word_values'])

    # step 2 Make a stratified sample of the classes and split into test and train sets


    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in split.split(dataframe_all['word_values'], dataframe_all['document_label']):
        strat_train_set = dataframe_all.loc[train_index]
        strat_test_set = dataframe_all.loc[test_index]

    strat_train_set = strat_train_set.dropna(subset=['word_values'])
    strat_test_set = strat_test_set.dropna(subset=['word_values'])

    x_train, y_train = su.pipe(strat_train_set), label_encoder.fit_transform(
        strat_train_set['document_label'])
    x_test, y_test = su.pipe(strat_test_set), label_encoder.fit_transform(
        strat_test_set['document_label'])

    return x_train,x_test,y_train,y_test


