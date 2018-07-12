# Models
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoCV, SGDRegressor, \
    LinearRegression
# Custom metric
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
import utils
import model


def cv(model, features, targets, folds=5):
    """Run cross validation for model object.

    Args:
        model (model): Model object.
        features (DataFrame): DataFrame with features
        targets (DataFrame): DataFrame with targets
        folds (int): Number of folds

    Returns:
        float: The CV score, in this case RMSLE
    """

    scorer = make_scorer(rmsl_metric, greater_is_better=True)
    cv_results = cross_validate(model.model, utils.drop_features(features, model.ignore_features), targets,
                                return_train_score=False, cv=folds, scoring=scorer)
    score = np.mean(cv_results["test_score"])
    print("CV score: {}".format(score))
    return score


def create_model(X_train, y_train, ignore_features=["Id"], type="GradientBoosting", X_validation=None,
                 y_validation=None, verbose=0):

    """Create model.

    Args:
        X_train (DataFrame): DataFrame with training features
        y_train (DataFrame): DataFrame with training targets
        ignore_features (list, optional): A list (str) with features to exclude from DataFrame
        type  (str, optional): Model type.
        X_validation (DataFrame, optional): DataFrame with validation features (e.g. for NN training; deprecated)
        y_validation (DataFrame, optional): DataFrame with validation targets (e.g. for NN training; deprecated)
        verbose (int, optional): print verbose output
    Returns:
        model: A custom model object
    """

    if "Id" not in ignore_features:
        ignore_features.append("Id")
        print("Ignoring Id field")
    X_train = X_train.drop(ignore_features, axis=1)
    if not X_validation is None:
        X_validation = X_validation.drop(ignore_features, axis=1)
    n_features = X_train.shape[1]

    learner = create_learner(type, n_features=n_features, X_validation=X_validation, y_validation=y_validation)

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
    """Create sklearn learner. To do: allow learner parameters to be passed in.

    Args:
        type  (str): Model type.
        n_features (int): Number of features in training data
        X_validation (DataFrame, optional): DataFrame with validation features (e.g. for NN training; deprecated)
        y_validation (DataFrame, optional): DataFrame with validation targets (e.g. for NN training; deprecated)

    Returns:
        learner: A Sklearn learner object
    """


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
        learner = ElasticNetCV(alphas=[1, .5, 0.1, .005, 0.001, 0.0005, .0001], max_iter=300, verbose=1, tol=.0001) #l1_ratio=.3,
    elif type == "ElasticNet":
        learner = ElasticNet(alpha=0.0005, l1_ratio=.3, max_iter=1000, tol=.0001)  # l1_ratio=0 means L2 error
    elif type == "OLS":
        learner = LinearRegression()
    elif type == "SGD":
        learner = SGDRegressor(random_state=0)
    elif type == "Lasso":
        learner = LassoCV(alphas=[1, .9, .5, 0.1, .005, 0.001, 0.0005, .0001], tol=0.0005)
    elif type == "SVR":
        learner = SVR(kernel='poly', C=1e3, verbose=0)
    else:
        print("Error, chosen model was not an option")
        return None
    return learner

def run_full_model(my_model, X_train, y_train, X_test, y_test=None, ignore_features=["Id"], exclude_misc=False):
    """Run model, prepare summary of results

    Args:
        my_model  (model): Model object.
        X_train (DataFrame): DataFrame with training features
        y_train (DataFrame): DataFrame with training targets
        X_test (DataFrame): DataFrame with test features
        y_test (DataFrame, optional): Dataframe with test targets
        ignore_features (list): List (str) of features to ignore
        exclude_misc (bool): If True, Misc_Val is being excluded from model and re-added post prediction

    Returns:
        dict: Has keys model (model), train_predictions
    """

    return_dict = {}
    submission_df = ""
    predicted_train = my_model.predict(X_train)
    predicted_test = my_model.predict(X_test)

    if exclude_misc:
        predicted_test["SalePrice"] += X_test["MiscVal"]

    if not y_test is None:
        pass # evaluate test data against known test values
    else:
        submission_df = pd.concat([X_test["Id"],pd.DataFrame(predicted_test, columns=["SalePrice"])], axis=1)

    submission_df = pd.concat([X_test["Id"], pd.DataFrame(predicted_test, columns=["SalePrice"])], axis=1)
    return_dict["model"] = my_model
    return_dict["train_predictions"] = predicted_train
    return_dict["test_predictions"] = predicted_test
    return_dict["submission_df"] = submission_df
    return return_dict

def rmsl_metric(ground_truth, predictions, take_log=False):
    """Root Mean Squared (Logarithmic) Error: Metric for evaluating model predictive power.

    Args:
        ground_truth  (DataFrame): Path to training data.
        path_test (DataFrame): Path to test data.
        take_log (bool, optional): Whether log should be taken of DataFrame values

    Returns:
        float: RMSLE
    """
    if not take_log:
        return np.sqrt(np.mean((predictions - ground_truth) ** 2))
    else:
        return np.sqrt(np.mean((np.log(1 + predictions) - np.log(1 + ground_truth)) ** 2))

if __name__ == '__main__':
    pass