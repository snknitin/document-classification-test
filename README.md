# HeavyWater Machine Learning Problem

### Methods Tried

Used micro averaging for F1 scores of all class labels

Logistic Regression(Multinomial)
    
    F1 score for training set: 0.9626
    F1 score for test set: 0.8879

XGBoost

    F1 score for training set: 0.8815
    F1 score for test set: 0.8755
    
Automatically adjusts to multinomial classification. Takes long to train and perfrm grid search for the best parameters, so abandoned due to tme constraint
    
Random Forest

    F1 score for training set: 0.9947
    F1 score for test set: 0.8546

Clearly overfit the training set. Will require pruning or rducing the max_depth to regularize
    
SVC
    
    F1 score for training set: 
    F1 score for test set: 
    
 Takes extremely long to train

I have only deployed using a pickled logistic regresion classifier since I was running out of time for proper hyperparameter tuning for the rest `classifier_LC.pkl`
I used a TfIdf Vectorizer in a pipeline since the tokens are all hashed.I stored this in `tfidf.pkl`.The pickle files help load the parameters during the deployment in production



### Data Engineering

Step-by-step exploration of the data and feature engineering is presented in the `Data_exploration` python notebook

- Read the data into a DataFrame
- Removed entries that had Nans
- Observed 14 different document labels
- Encoded these labels to make categorical values discrete
- Transformed token entries in each documentto the tfidf scores
- Created a Stratified Sample train/test split since class distribution is not even


### Deployment

Using Flask for REST based api calls

    python runner.py

will enable the service on the localhost='127.0.0.1' on port=4000. The service requests a document's content hashed in the same way the dataset was. Clicking predict button will display which class the document belongs to. 

Pushing it to AWS Lambda using Zappa and aws-cli to becoem a serverless function

    https://cw48d6z2bb.execute-api.us-west-2.amazonaws.com/dev


