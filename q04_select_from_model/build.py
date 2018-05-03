# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    X = df.drop(labels=['SalePrice'], axis=1)
    y = df.SalePrice
    rf = RandomForestClassifier(random_state=9)
    rf.fit(X,y)
    # prefit is set to True since we have already fitted a model
    feat_selector = SelectFromModel(estimator=rf,prefit=True)
    # This gives the array of all the features and selected feature indices are marked as True
    feat_selector_arr = feat_selector.get_support()
    # Get the indices which are marked as True from the above array
    ind = np.where( feat_selector_arr == True )
    # Select the indices from the feature columns passed to the selectfrommodel
    sel_features = X.columns[ind].values.tolist()
    return sel_features
