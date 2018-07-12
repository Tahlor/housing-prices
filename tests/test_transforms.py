"""Tests the transform functions defined in transforms.py
"""

import numpy as np
import pandas as pd

import transforms
import utils

feats_with_id = pd.DataFrame({'col1': [3, 4], 'Id': [1, 2]})
feats = pd.DataFrame(np.asarray([3, 4]))
targs = pd.DataFrame(np.asarray([1, 6]))

model_list = ["OLS"]

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

d_missing = {'LotFrontage': [1, np.NaN], 'col2': [3, 4]}
df_missing = pd.DataFrame(data=d_missing)

# All of these tests should be expanded to test all possible transformations, including edge cases (0's, negatives, NaNs, etc.)

def test_transform():
    new_df, _ = transforms.transform(df, rename = True, replace = False, trans_type = "quadratic", scaler = None, symmetric = None, verbose = 0)
    assert utils.checkEqual(new_df["col1_trans_quadratic"].values, [1,4])
    assert utils.checkEqual(new_df["col2_trans_quadratic"].values, [9,16])

def test_perform_operations():
    new_df = transforms.perform_operations(df, "subtract", symmetric=False, verbose=0)
    assert utils.checkEqual(new_df["col1_subtract_col2"].values, [-2,-2])
    assert utils.checkEqual(new_df["col2_subtract_col1"].values, [2,2])

def test_perform_single_op():
    new_df = transforms.perform_single_op(df['col1'], df['col2'], "multiply", "new_column", verbose = 0)
    assert utils.checkEqual(new_df.values, [3, 8])

def test_rescale():
    rescale_df, _ = transforms.rescale(df, scaler = None, scaling_type= "normal")
    assert utils.checkEqual(rescale_df["col1"].values, [-1,1])
    assert utils.checkEqual(rescale_df["col2"].values, [-1,1])

if __name__=='__main__':
    test_transform()
    test_perform_operations()
    test_perform_single_op()
    test_rescale()