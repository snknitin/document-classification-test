import os
import matplotlib.pyplot as plt
import numpy as np

IMAGES_PATH=os.path.join(os.getcwd(),"static/")
if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
label_encoder = LabelEncoder()
# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class LabelEncoderPipelineFriendly(LabelEncoder):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelEncoderPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelEncoderPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelEncoderPipelineFriendly, self).fit(X).transform(X)

class ArrayCaster(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.transpose(np.matrix(data))

def pipe():
    num_pipeline = Pipeline([
            ('selector', DataFrameSelector('word_values')),
            ('vect', TfidfVectorizer(max_df=0.95, min_df=2, max_features=1037929)),
            ])

    cat_pipeline = Pipeline([
            ('selector', DataFrameSelector('document_label')),
            ('encoder', LabelEncoderPipelineFriendly()),
        ('caster', ArrayCaster()),
        ])

    from sklearn.pipeline import FeatureUnion

    full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
              #("cat_pipeline", cat_pipeline),
        ])
    return full_pipeline

