"""Tests the import function defined in prep_features.py
"""

import housing_prices
from housing_prices import prep_features
import pandas as pd
import utils
import numpy as np

train_path = r'../data/train.csv'

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

d_missing = {'LotFrontage': [1, np.NaN], 'col2': [3, 4]}
df_missing = pd.DataFrame(data=d_missing)

real_df = housing_prices.import_data.process_data(train_path)

def test_import():
    train_path = r'../../data/train.csv'
    assert real_df.shape == (1460, 81)

def test_create_market_index():
    train_path = r'../data/train.csv'
    real_df['TotalSQF'] = real_df['TotalBsmtSF'] + real_df['GrLivArea']
    index = prep_features.create_market_index(real_df, index_name="time", vars=["MoSold", "YrSold"])
    assert (index["time_index"] > .5).all() and (index["time_index"] < 1.5).all()

    # Assert keys unqiuely identify rows
    #df1[df1.duplicated(subset=["Neighborhood"], keep=False)]

def test_feature_standardization():
    # Feature standardization depends on several other functions

    pd.options.mode.chained_assignment = None  # default='warn'

    def test_recode_features():
        return prep_features.recode(real_df)

    def test_create_new_features(df):
        new_feats = prep_features.create_new_features(df)
        return new_feats

    feats = test_recode_features()
    feats,_,_ = test_create_new_features(feats)
    feats, targs = prep_features.feature_standardization(feats)
    utils.checkEqual(targs.columns.values[:], ["SalePrice]", "SalePriceMiscVal"])


def test_recode_features():
    d = {1:9, 2:9}
    features = prep_features.recode_features(df, "col1", d, missing=0)
    assert (features['col1']==9).all()


def test_get_features():
    feats = prep_features.get_features(df, "1")
    assert feats == ["col1"]

# smart_impute - may be deprecated

def test_check_for_missing():
    error = ""
    try:
        prep_features.check_for_missing(df_missing)
    except Exception as ex:
        error = ex
    assert error.__class__.__name__ == "AssertionError"

def test_impute_missings():
    imputed_df = prep_features.impute_missings(df_missing)
    prep_features.check_for_missing(imputed_df)

def test_specific_transform():
    d = {'var1': [1, 2], 'var2': [3, 4]}
    df = pd.DataFrame(data=d)
    col = "var1_trans_quadratic_multiply_var2_trans_cubic_multiply"
    result_col = prep_features.specific_transform(df, col, verbose=0)["var1_trans_quadratic_multiply_var2_trans_cubic_trans_multiply"]
    utils.checkEqual(result_col.values, [27, 256])

def test_create_vacuous_variables():
    # Make sure all features in reference dataframe are found in the final one
    new_df = prep_features.create_vacuous_variables(df, df_missing)
    assert all(x in new_df.columns for x in df)

def test_main():
    # Master test
    feats,targs,_,_ = prep_features.main(real_df, real_df)

    # Check shapes
    assert feats.shape[0] == 1460
    assert feats.shape[1] > 400
    assert targs.shape == (1460, 2)

    # Make sure features made it in as expected
    assert "TotalSQF_orig_cont" in feats.columns.values

    # Make sure SalePrice didn't sneak into features
    assert all("SalePrice" not in x for x in feats.columns.values)

if __name__=='__main__':
    test_import()
    test_create_market_index()
    test_recode_features()
    test_get_features()
    test_feature_standardization()
    test_check_for_missing()
    test_impute_missings()
    test_specific_transform()
    test_create_vacuous_variables()
    test_main()