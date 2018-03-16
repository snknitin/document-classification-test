# HeavyWater Machine Learning Problem

### Methods Tried

Logistic Regression(Multinomial)
    F1 score for training set: 0.9626

    F! score for test set: 0.8879
XGBoost
Random Forest
SVC

F1 Scores



I have only deployed using a pickled logistic regresion classifier since I was running out of time for proper hyperparameter tuning for the rest

classifier_LC.pkl

I used a TfIdf Vectorizer in a pipeline since the tokens are all hashed.

I stored this in tfidf.pkl



### Data Engineering

Step-by-step exploration of the data and feature engineering is presented in the Data_exploration python notebook

- Read the data into a DataFrame
- Removed entries that had Nans
- Observed 14 different document labels
- Encoded these labels to make categorical values discrete
- Transformed token entries in each documentto the tfidf scores
- Created a Stratified Sample train/test split since class distribution is not even


### Deployment

Using Flask for REST based api calls

python runner.py

will enable the service on the localhost='127.0.0.1' on port=4000

Pushing it to AWS Lambda using Zappa and aws-cli


