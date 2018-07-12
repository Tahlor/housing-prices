from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Models
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, LassoCV, LassoLarsCV, SGDRegressor, \
    LinearRegression
from sklearn.svm import SVR
from xgboost import plot_importance

# Performance metrics
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate, cross_val_score

# NN stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

# Custom metric
from sklearn.metrics import make_scorer

import pandas as pd
import numpy as np
import model

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import math


def cv(model, features, targets):
    scorer = make_scorer(rmsl_metric, greater_is_better=True)
    cv_results = cross_validate(model.model, drop_features(features, model.ignore_features), targets,
                                return_train_score=False, cv=5, scoring=scorer)
    score = np.mean(cv_results["test_score"])
    print("CV score: {}".format(score))
    return score


def drop_features(features, ignore_features):
    for f in ignore_features:
        if f in features.columns:
            features.drop(f, inplace=True, axis=1)
        else:
            print("{} not found in features".format(f))
    return features


def create_model(X_train, y_train, ignore_features=["Id"], type="GradientBoosting", X_validation=None,
                 y_validation=None, verbose=0):
    if "Id" not in ignore_features:
        ignore_features.append("Id")
        print("Ignoring Id field")
    X_train = X_train.drop(ignore_features, axis=1)
    if not X_validation is None:
        X_validation = X_validation.drop(ignore_features, axis=1)
    n_features = X_train.shape[1]

    learner = create_learner(type, n_features=n_features, X_validation=X_validation, y_validation=y_validation)

    if type == "NN":
        learner.fit(X_train, y_train)
    else:
        learner.fit(X_train, y_train)
    if type in ['Lasso', 'ElasticNet']:
        coef = pd.Series(learner.coef_, index=X_train.columns)
        non_zero_indices = list(np.where(coef != 0)[0])
        variables_chosen = X_train.columns[non_zero_indices]
        if verbose:
            print("Alpha {}".format(learner.alpha_))
            print(variables_chosen)
            print("Kept {} variables".format(len(non_zero_indices)))
    my_model = model.model(ignore_features, learner, learner_type=type)
    return my_model


def create_learner(type, n_features=None, X_validation=None, y_validation=None):
    if type == "RandomForest":
        params = {'n_estimators': 30, 'oob_score': False, 'random_state': 0, 'max_features': int(n_features / 5),
                  'max_depth': None, 'min_samples_split': 2, 'bootstrap': True, 'min_samples_leaf': 1}
        learner = RandomForestRegressor(**params)
    elif type == "GradientBoosting":
        # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
        #           'learning_rate': 0.01, 'loss': 'ls'}
        params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.05, 'loss': 'ls', 'verbose': 0, 'subsample': .7}
        learner = GradientBoostingRegressor(**params)
    elif type == "GradientBoostingProduction":
        params = {'n_estimators': 2000, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls', 'verbose': 0, 'subsample': .6}
        learner = GradientBoostingRegressor(**params)
    elif type == "XGBoosting":
        params = {'n_estimators': 200, 'colsample_bytree': 0.9083255949777539, 'subsample': 0.9875583602680632,
                  'min_child_weight': 3.3691725891936293, 'max_depth': 12, 'learning_rate': 0.1979044659137234,
                  'gamma': 6.190441877060026, 'reg_alpha': 17.9555343985871}
        learner = XGBRegressor(**params)
    elif type == "ElasticNetCV":
        learner = ElasticNetCV(alphas=[1, .5, 0.1, .005, 0.001, 0.0005, .0001], max_iter=20000, verbose=1, tol=100000) #l1_ratio=.3,
    elif type == "ElasticNet":
        learner = ElasticNet(alpha=0.0005, l1_ratio=.3)  # l1_ratio=0 means L2 error
    elif type == "OLS":
        learner = LinearRegression()
    elif type == "SGD":
        learner = SGDRegressor(random_state=0)
    elif type == "Lasso":
        learner = LassoCV(alphas=[1, .9, .5, 0.1, .005, 0.001, 0.0005, .0001], tol=0.0005)
    elif type == "Voting":
        learners = [create_learner("GradientBoosting"), create_learner("Lasso"), create_learner("ElasticNet")]
    elif type == "SVR":
        # learner = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=1)
        # learner = SVR(kernel='linear', C=1e3, verbose=1)
        learner = SVR(kernel='poly', C=1e3, verbose=1)
    else:
        print("Error, chosen model was not an option")
    return learner


def run_full_model(my_model, X_train, y_train, X_test, y_test=None, ignore_features=["Id"], exclude_misc=False):
    return_dict = {}
    submission_df = ""
    predicted_train = my_model.predict(X_train)
    predicted_test = my_model.predict(X_test)

    if exclude_misc:
        predicted_test["SalePrice"] += X_test["MiscVal"]

    # Compute scores only if y-test is provided
    if not y_test is None:
        evaluate_model(my_model, predicted_test, y_test)
    else:
        submission_df = pd.concat([X_test["Id"], pd.DataFrame(predicted_test, columns=["SalePrice"])], axis=1)
    return_dict["model"] = my_model
    return_dict["train_predictions"] = predicted_train
    return_dict["test_predictions"] = predicted_test
    return_dict["submission_df"] = submission_df
    return return_dict

def evaluate_model(model_obj, features, labels):
    predictions = model_obj.predict(features)
    test_score = r2_score(predictions, labels)
    spearman = spearmanr(predictions, labels)
    pearson = pearsonr(predictions, labels)

    # rms = sqrt(mean_squared_error(y_test, predicted_test))
    rmsl = rmsl_metric(labels, predictions)
    print('Out-of-bag R-2 score estimate: {}'.format(model.oob_score_))
    print('Test data R-2 score: {}'.format(test_score))
    print('Test data Spearman correlation: {}'.format(spearman[0]))
    print('Test data Pearson correlation: {}'.format(pearson[0]))
    print('RMSL: {}'.format(rmsl))
    return rmsl


def rmsl_metric(ground_truth, predictions, take_log=False):
    if not take_log:
        return np.sqrt(np.mean((predictions - ground_truth) ** 2))
    else:
        return np.sqrt(np.mean((np.log(1 + predictions) - np.log(1 + ground_truth)) ** 2))


def split_train_test(features, targets, train_size=1):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=1, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_features(model, input_features, verbose=0):
    learner_type = model.learner_type
    input_features = input_features.drop(model.ignore_features, 1)
    model = model.model
    # print(learner_type)
    if learner_type in ["RandomForest", "GradientBoosting", "XGBoosting", "GradientBoostingProduction"]:
        importances = model.feature_importances_
        feature_list = zip(importances, input_features)

    elif learner_type in ["Lasso", "ElasticNet", "OLS", "ElasticNetCV"]:
        coef = pd.Series(model.coef_, index=input_features.columns)
        non_zero_indices = list(np.where(coef != 0)[0])
        variables_chosen = input_features.columns[non_zero_indices].values
        coefs_chosen = coef[non_zero_indices].values
        # print(len(variables_chosen))
        # print(len(coefs_chosen))

        feature_list = zip(coefs_chosen * 100, variables_chosen)
    feature_list = list(feature_list)
    feature_list.sort(key=lambda x: -abs(x[0]))
    if verbose:
        for x in feature_list:
            print(x)
    return feature_list


def feature_tree(my_model):
    import numpy as np
    import matplotlib.pyplot as plt
    my_model = my_model.model
    importances = my_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in my_model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the my_model
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


if __name__ == '__main__':
    pass