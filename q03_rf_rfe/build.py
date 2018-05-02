# Default imports
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    X = df.drop(labels=['SalePrice'], axis=1)
    y = df.SalePrice
    no_of_features = X.shape[1]
    features_to_select = no_of_features / 2
    rf = RandomForestClassifier(random_state=9)
    rfe = RFE(estimator=rf, n_features_to_select=features_to_select)
    X_transformed = rfe.fit_transform(X,y)
    # print X_transformed.shape
    feat_ranking = rfe.ranking_
    # Get indices of the features which are ranked 1 (selected)
    ind = np.where(feat_ranking == 1)
    # Get the actual feature names based on indices
    lst = X.columns[ind].values.tolist()
    return lst
