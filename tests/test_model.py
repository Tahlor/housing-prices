"""Tests the import function defined in import_data.py
"""

import numpy as np
import pandas as pd

from housing_prices import import_data
from housing_prices import model_functions
from housing_prices import prep_features
import os
dirname = os.path.dirname(__file__)
test_path  = os.path.join(dirname, r'test_data/test.csv')
train_path = os.path.join(dirname, r'test_data/train.csv')

feats_with_id = pd.DataFrame({'col1': [3, 4], 'Id': [1, 2]})
feats = pd.DataFrame(np.asarray([3, 4]))
targs = pd.DataFrame(np.asarray([1, 6]))
model_list = ["GradientBoosting", "GradientBoostingProduction", "ElasticNet", "RandomForest", "XGBoosting", "SGD", "ElasticNetCV", "Lasso", "SVR", "OLS",
              "NotARealModel"]

def create_model():
    model = model_functions.create_model(feats_with_id, targs, type="ElasticNet")
    return model

def test_rmsl_metric():
    score = model_functions.rmsl_metric(feats, targs, False).values[0]
    assert score == 2

def test_cv():
    model = create_model()
    score = model_functions.cv(model, feats_with_id, targs, folds=2)
    assert score ==5

def test_run_full_model():
    m = create_model()
    d = model_functions.run_full_model(m, feats_with_id, targs, feats_with_id, y_test=None, ignore_features=["Id"], exclude_misc=False)

    assert all(x in d.keys() for x in ["train_predictions", "model", "test_predictions", "submission_df"])
    assert ["Id"] in d["submission_df"].columns.values
    assert d["submission_df"].shape == (2,2)
    assert d["train_predictions"].shape == (feats_with_id.shape[0],)

# Full test of models
def test_models():

    keep_variables = ['1stFlrSF_cont_trans_log', 'Neighborhood_MeadowV_orig_categ', 'LotArea_cont_trans_log',
                      'Neighborhood_Crawfor_orig_categ', 'BsmtFinType2_orig_ord', 'MSZoning_FV_orig_categ',
                      'Neighborhood_StoneBr_orig_categ', 'Heating_Grav_orig_categ', 'HouseAge_cont_trans_normal',
                      'GarageIdx_cont_trans_root', 'Neighborhood_Edwards_orig_categ',
                      'CentralAir_ord_trans_normal_multiply_Neighborhood_Crawfor_categ_trans_multiply',
                      'WoodDeckSF_cont_trans_scale01', 'CentralAir_ord_trans_inverse',
                      'SaleCondition_Family_orig_categ',
                      'GarageFinish_ord_trans_quadratic_multiply_MSZoning_C (all)_categ_trans_multiply',
                      'MSSubClass_30_orig_categ',
                      'OverallCond_cont_trans_quadratic_divide_OverallQual_cont_trans_quadratic_trans_divide',
                      'Neighborhood_OldTown_orig_categ', 'BsmtFinSF1_cont_trans_normal', 'MSSubClass_90_orig_categ',
                      'Exterior2nd_Stucco_orig_categ', 'LandContour_HLS_orig_categ',
                      'CentralAir_ord_trans_normal_multiply_MSZoning_RL_categ_trans_multiply',
                      'TotRmsAbvGrd_cont_trans_normal',
                      '2ndFlrSF_cont_trans_root_divide_SQFperRoom_cont_trans_root_trans_divide',
                      'GrLivArea_cont_divide_GrLivArea_cont_trans_root_trans_divide', 'MSZoning_RL_orig_categ',
                      'BldgType_1Fam_orig_categ',
                      'neighborhood_index_cont_trans_normal_divide_CentralAir_ord_trans_normal_trans_divide',
                      'Foundation_BrkTil_orig_categ', 'LotConfig_FR2_orig_categ', 'Fireplaces_cont_trans_normal',
                      'MSSubClass_120_orig_categ',
                      '1stFlrSF_cont_trans_root_divide_HeatingQC_ord_trans_quadratic_trans_divide',
                      'CentralAir_ord_trans_normal_multiply_HalfBath_cont_trans_normal_trans_multiply',
                      'PoolQC_ord_trans_quadratic',
                      'TotalBsmtSF_cont_trans_root_divide_HeatingQC_ord_trans_quadratic_trans_divide',
                      'BldgType_Twnhs_orig_categ', 'RemodelAge_cont_trans_log', 'Foundation_PConc_orig_categ',
                      'MasVnrArea_cont_trans_normal', 'GarageType_BuiltIn_orig_categ',
                      'ScreenPorch_cont_trans_normal',
                      'SaleCondition_Abnorml_categ_multiply_GarageFinish_ord_trans_quadratic_trans_multiply',
                      'PavedDrive_ord_trans_quadratic',
                      'OverallCond_cont_trans_normal_multiply_HouseAge_cont_trans_root_trans_multiply',
                      'BsmtUnfSF_cont_trans_normal', 'BsmtFinType1_ord_trans_quadratic',
                      'HouseStyle_1.5Fin_orig_categ', 'LotShape_IR1_orig_categ', 'YrSold_2009_orig_categ',
                      'BedroomAbvGr_cont_trans_normal',
                      'GrLivArea_cont_trans_root_divide_BsmtFinType2_ord_trans_quadratic_trans_divide',
                      'YrSold_2010_orig_categ', 'Exterior2nd_Wd Sdng_orig_categ', 'RoofStyle_Hip_orig_categ',
                      'GarageIdx_cont_multiply_OverallCond_cont_trans_multiply', 'BsmtExposure_ord_trans_quadratic',
                      'LotFrontage_cont_multiply_SaleCondition_Abnorml_categ_trans_multiply',
                      'OverallCond_cont_trans_quadratic_multiply_Condition1_Norm_categ_trans_multiply',
                      'LowQualFinSF_cont_trans_root',
                      'BsmtFullBath_cont_trans_normal_multiply_BsmtExposure_ord_trans_quadratic_trans_multiply',
                      'FullBath_cont_trans_quadratic_multiply_KitchenQual_ord_trans_quadratic_trans_multiply',
                      'OverallCond_cont_trans_quadratic_multiply_OverallCond_cont_trans_normal_trans_multiply',
                      'BsmtUnfSF_cont_trans_root_multiply_OverallCond_cont_trans_normal_trans_multiply',
                      'OverallQual_cont_trans_quadratic_multiply_GarageQual_ord_trans_quadratic_trans_multiply',
                      'Functional_ord_trans_quadratic_multiply_OverallCond_cont_trans_multiply',
                      '1stFlrSF_cont_trans_root_multiply_GarageIdx_cont_trans_multiply',
                      'Functional_ord_trans_quadratic_multiply_SummerSale_categ_trans_multiply',
                      'OverallQual_cont_trans_quadratic_multiply_KitchenAbvGr_cont_trans_normal_trans_multiply',
                      'OverallQual_cont_trans_quadratic_multiply_BsmtFullBath_cont_trans_normal_trans_multiply',
                      'GarageCars_cont_trans_quadratic_multiply_TotalBsmtSF_cont_trans_root_trans_multiply',
                      'OverallQual_cont_trans_quadratic_multiply_GarageIdx_cont_trans_multiply',
                      'MasVnrArea_cont_trans_root_multiply_GrLivArea_cont_trans_root_trans_multiply',
                      'Exterior1st_BrkFace_categ_multiply_SQFperRoom_cont_trans_multiply',
                      'MasVnrArea_cont_trans_root_multiply_Functional_ord_trans_quadratic_trans_multiply',
                      'GrLivArea_cont_multiply_neighborhood_index_cont_trans_normal_trans_multiply',
                      'neighborhood_index_cont_trans_normal_multiply_MasVnrArea_cont_trans_multiply',
                      'neighborhood_index_cont_trans_normal_multiply_TotalBsmtSF_cont_trans_multiply',
                      'GarageArea_cont_trans_root_multiply_OverallCond_cont_trans_normal_trans_multiply',
                      'TotRmsAbvGrd_cont_trans_quadratic_multiply_GarageIdx_cont_trans_quadratic_trans_multiply',
                      'GrLivArea_cont_multiply_GarageIdx_cont_trans_multiply',
                      'OverallCond_cont_trans_quadratic_multiply_LotArea_cont_trans_root_trans_multiply',
                      '1stFlrSF_cont_trans_quadratic_divide_TotalSQF_cont_trans_root_trans_divide',
                      'SQFperRoom_cont_trans_quadratic',
                      'TotalSQF_cont_multiply_MasVnrType_ord_trans_quadratic_trans_multiply',
                      'EnclosedPorch_cont_trans_quadratic',
                      'TotalSQF_cont_trans_quadratic_divide_LotArea_cont_trans_root_trans_divide',
                      'SQFperRoom_cont_trans_quadratic_multiply_GarageQual_ord_trans_quadratic_trans_multiply',
                      '2ndFlrSF_cont_trans_root_multiply_HouseAge_cont_trans_quadratic_trans_multiply',
                      'BsmtFinSF1_cont_trans_quadratic_divide_BsmtFinType1_ord_trans_quadratic_trans_divide',
                      'Exterior1st_BrkFace_categ_multiply_GrLivArea_cont_trans_quadratic_trans_multiply',
                      'TotalBsmtSF_cont_trans_quadratic',
                      'BsmtFinSF1_cont_trans_quadratic_multiply_Exterior1st_BrkFace_categ_trans_multiply',
                      'GrLivArea_cont_trans_quadratic_divide_BsmtFinType1_ord_trans_quadratic_trans_divide',
                      'GarageArea_cont_trans_quadratic_multiply_PrchSQ_cont_trans_root_trans_multiply',
                      'TotalSQF_cont_trans_quadratic_divide_CentralAir_ord_trans_normal_trans_divide',
                      'OpenPorchSF_cont_trans_root_multiply_BsmtUnfSF_cont_trans_quadratic_trans_multiply',
                      'GarageIdx_cont_multiply_GrLivArea_cont_trans_quadratic_trans_multiply',
                      '2ndFlrSF_cont_trans_quadratic_multiply_1stFlrSF_cont_trans_root_trans_multiply',
                      'GarageIdx_cont_multiply_LotArea_cont_trans_quadratic_trans_multiply', 'Id']

    training_df = import_data.process_data(train_path)
    test_df = import_data.process_data(test_path)
    features, targets, test_features, _ = prep_features.main(training_df, test_df, use_log=True, variable_combinations=keep_variables)
    targets = targets.drop(["SalePriceMiscVal"], axis=1)
    features = features[keep_variables]
    test_features = features[keep_variables]

    # Prep targets
    targets = targets["SalePrice"]

    for model in model_list:
        print("Running {}".format(model))
        my_model = model_functions.create_model(features, targets, type=model)
        model_functions.cv(my_model, features, targets)

if __name__=='__main__':
    #test_create_learner()
    #test_rmsl_metric()
    #test_cv()
    #test_run_full_model()
    test_models()

