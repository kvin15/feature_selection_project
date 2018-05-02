# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df, k=20):
    X = df.drop(labels=['SalePrice'], axis=1)
    y = df.SalePrice
    feat_selector = SelectPercentile(score_func=f_regression, percentile=k)
    X_transformed = feat_selector.fit_transform(X,y)
    # print '(Before f_regression and SelectKPercentile) Predictor rows {} and columns {}'.format(X.shape[0],X.shape[1])
    # print '(After f_regression and SelectKPercentile) Predictor rows {} and columns {}'.format(X_transformed.shape[0],X_transformed.shape[1])
    # print 'Selected {} columns post the f_regression and SelectKPercentile'.format(X_transformed.shape[1])
    sel_params_index = feat_selector.get_support(indices=True)
    sel_params_scores = feat_selector.scores_[sel_params_index]
    sel_params = X.columns[sel_params_index].values
    # Get the indices as per the scores sorted in descending order
    sort_indices = sel_params_scores.argsort()[::-1]
    # Using the above indices sort the columns/features selected by our SelectPercentile
    lst = sel_params[sort_indices].tolist()
    return lst
