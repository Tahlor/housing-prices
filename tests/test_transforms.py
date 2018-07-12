"""Tests the import function defined in import_data.py
"""

from housing_prices import model_functions
from housing_prices import main
from housing_prices import import_data
from housing_prices import prep_features

import pandas as pd
import numpy as np
import transforms

feats_with_id = pd.DataFrame({'col1': [3, 4], 'Id': [1, 2]})
feats = pd.DataFrame(np.asarray([3, 4]))
targs = pd.DataFrame(np.asarray([1, 6]))

model_list = ["OLS"]


def test_transform():
    transforms.transform(features, rename = True, replace = False, trans_type = None, scaler = None, symmetric = None, verbose = 0)

def test_perform_operations():
    transforms.perform_operations(df, op, symmetric=None, verbose=0):


def test_perform_single_op():
    transforms.perform_single_op(df1, df2, op, col_name, verbose = 0):

def test_rescale():
    transforms.rescale(features, scaler = None, scaling_type= None):



if __name__=='__main__':
    #test_drop_features()
    #test_create_learner()
    #test_rmsl_metric()
    #test_cv()
    #test_run_full_model()
    test_models()



