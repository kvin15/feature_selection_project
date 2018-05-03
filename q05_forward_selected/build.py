# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


# Your solution code here
def forward_selected(data, model):
    # final output variables
    # output variable feature list contains features in
    # the order of their exploration and addition
    feat_list = list()
    # output variable r2_score contains r2_scores for features in
    # the increasing order. The element at index 1 (second element)
    # indicates r2_score for (feature-1, feature-2) in above feature_list
    # element at index 2 (third element) indicates r2_score
    # (feature-1,feature-2,feature-3) in the above feature list
    r2_score_feat = list()
    X = data.drop(labels=['SalePrice'], axis=1)
    y = data.SalePrice
    no_of_predictors = X.shape[1]
    predictor_lst_orig = X.columns.values.tolist()
    # iteration variables and configurations
    explore_more = True
    # the below list keeps track of features remaining to be
    # explored / combined and is updated after every iteration
    predictor_lst_update = list(predictor_lst_orig)
    iteration = 0
    while explore_more:
        r2_score_tmp = list()
        # for iteration for checking which feature when combined
        # gives the best result for the above iteration
        for val in predictor_lst_update:
            cols_to_fetch = list()
            if len(feat_list) == 0:
                cols_to_fetch.append(val)
            else:
                cols_to_fetch = list(feat_list)
                cols_to_fetch.append(val)
            X_feat_new = X.loc[:,cols_to_fetch]
            model.fit(X_feat_new,y)
            r2_score_test = model.score(X_feat_new,y)
            r2_score_tmp.append(r2_score_test)
        # close of for loop and we check max scores of combination
        # of which remaining feature with previously (in earlier
        # iteration) selected feature
        max_score_tmp = max(r2_score_tmp)
        max_index_tmp = r2_score_tmp.index(max_score_tmp)
        max_index_feature_tmp = predictor_lst_update[max_index_tmp]
        iteration += 1
        prev_max = -999
        if len(r2_score_feat) == 0:
            prev_max = -999
        else:
            prev_max = max(r2_score_feat)

        # check if score is maximum than previous iteration then
        # 1. Add the r2_score as well as feature to our output variables
        # 2. Remove the feature already explored from our exploration list/
        # exploration list is predictor_lst_update
        # set the iteration configuration variable explore_more to True
        # to continue iteration/exploration with the remaining variables
        if max_score_tmp > prev_max:
            explore_more = True
            feat_list.append(max_index_feature_tmp)
            r2_score_feat.append(max_score_tmp)
            # update the predictor list to remove the max index feature
            predictor_lst_update.remove(max_index_feature_tmp)
        else:
            explore_more = False

        if len(predictor_lst_update) == 0:
            explore_more = False
        # clode of while loop
    return feat_list, r2_score_feat
