import numpy as np
import pandas as pd
from sklearn.externals import joblib
import data_preprocess as dp
from flask import Flask, request,flash, jsonify, render_template, make_response
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField


DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    words = TextField('Words:', validators=[validators.required()])


@app.route("/")
def index():
    form = ReusableForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        words = request.form['words']
        print(words)

        if not form.validate():
            flash('All the form fields are required. ')


    return render_template('index.html',form=form)

@app.route("/",methods=['GET', 'POST'])
def predict():
    #Load model
    clf = joblib.load('classifier_LC.pkl')
    tfidf = joblib.load('tfidf.pkl')
    labels = ['APPLICATION', 'BILL', 'BILL BINDER', 'BINDER', 'CANCELLATION NOTICE', 'CHANGE ENDORSEMENT',
              'DECLARATION',
              'DELETION OF INTEREST', 'EXPIRATION NOTICE', 'INTENT TO CANCEL NOTICE', 'NON-RENEWAL NOTICE',
              'POLICY CHANGE',
              'REINSTATEMENT NOTICE', 'RETURNED CHECK']

    label_mapper = {k: v for k, v in enumerate(labels)}
    # Take data value and get features
    words = request.form['words']
    #print(words)
    words=pd.DataFrame([words])
    words.columns = ["word_values"]
    features=tfidf.transform(words["word_values"])
    print(features.shape)
    prediction=clf.predict(features)
    print(prediction)
    print(label_mapper[prediction[0]])
    return '<h1> Your document is of type {} '.format(label_mapper[prediction[0]])




if __name__ == '__main__':
    app.run(host='127.0.0.1', port=4000)


