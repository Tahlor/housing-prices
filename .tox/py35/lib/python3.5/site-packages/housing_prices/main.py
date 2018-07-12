import numpy as np
import import_data
import model_functions
import prep_features
import export_data
import numpy as np

import export_data
import import_data
import model_functions
import prep_features

MODEL_TYPE = "ElasticNet" # "GradientBoosting" "RandomForest" "XGBoosting" "SGD" "ElasticNet" "Lasso" "Voting" "SVR" "NN" "auto"
IGNORE = ["Id","MiscVal"]
EXC_MISC_VAL = False
if not EXC_MISC_VAL:
    IGNORE.remove("MiscVal")
USE_LOG = True

def main(path_training, path_test, training_test_seed=None, output_path = r"./output.csv", model_type=MODEL_TYPE, exclude_misc=EXC_MISC_VAL, keep_variables=None):
    """Processes data, predicts housing prices, saves predictions.

    Args:
        path_training  (str): Path to training data.
        path_test (str): Path to test data.
        training_test_seed (int, optional): Random seed for internal training/test split.
        output_path ( str): Location to save predicitions

    Returns:
        None
    """
    
    # Import Features
    training_df = import_data.process_data(path_training)
    test_df = import_data.process_data(path_test)

    # Preprocess features
    features, targets, test_features, _ = prep_features.main(training_df, test_df, use_log=USE_LOG, variable_combinations=keep_variables)

    # Optionally pull misc value out of sale price and add back in after
    if exclude_misc:
        targets = targets.drop(["SalePrice"], axis=1).rename(index=str, columns={"SalePriceMiscVal": "SalePrice"})
    else:
        targets = targets.drop(["SalePriceMiscVal"], axis=1)

    # Limit variables
    #print("Keep Variables not created")
    #print(set(keep_variables)-set(features.columns))
    #print("Created variabls not kept")
    #print(set(features.columns)-set(keep_variables))


    if not keep_variables is None:
        features = features[keep_variables]
        test_features = features[keep_variables]

    # Prep targets
    targets = targets["SalePrice"]
    my_model = model_functions.create_model(features, targets, ignore_features= IGNORE, type=model_type)
    model_functions.cv(my_model, features, targets)

    # Export results
    result_dict = model_functions.run_full_model(my_model, features, targets, test_features, exclude_misc=exclude_misc)
    cs_df = result_dict["submission_df"]

    if USE_LOG:
        cs_df["SalePrice"] = np.exp(cs_df["SalePrice"])-1
    export_data.export_data(cs_df, output_path)

if __name__=='__main__':
    test_path = r'../data/test.csv'
    train_path = r'../data/train.csv'
    keep_variables = ['1stFlrSF_cont_trans_log', 'Neighborhood_MeadowV_orig_categ', 'LotArea_cont_trans_log', 'Neighborhood_Crawfor_orig_categ', 'BsmtFinType2_orig_ord', 'MSZoning_FV_orig_categ', 'Neighborhood_StoneBr_orig_categ', 'Heating_Grav_orig_categ', 'HouseAge_cont_trans_normal', 'GarageIdx_cont_trans_root', 'Neighborhood_Edwards_orig_categ', 'CentralAir_ord_trans_normal_multiply_Neighborhood_Crawfor_categ_trans_multiply', 'WoodDeckSF_cont_trans_scale01', 'CentralAir_ord_trans_inverse', 'SaleCondition_Family_orig_categ', 'GarageFinish_ord_trans_quadratic_multiply_MSZoning_C (all)_categ_trans_multiply', 'MSSubClass_30_orig_categ', 'OverallCond_cont_trans_quadratic_divide_OverallQual_cont_trans_quadratic_trans_divide', 'Neighborhood_OldTown_orig_categ', 'BsmtFinSF1_cont_trans_normal', 'MSSubClass_90_orig_categ', 'Exterior2nd_Stucco_orig_categ', 'LandContour_HLS_orig_categ', 'CentralAir_ord_trans_normal_multiply_MSZoning_RL_categ_trans_multiply', 'TotRmsAbvGrd_cont_trans_normal', '2ndFlrSF_cont_trans_root_divide_SQFperRoom_cont_trans_root_trans_divide', 'GrLivArea_cont_divide_GrLivArea_cont_trans_root_trans_divide', 'MSZoning_RL_orig_categ', 'BldgType_1Fam_orig_categ', 'neighborhood_index_cont_trans_normal_divide_CentralAir_ord_trans_normal_trans_divide', 'Foundation_BrkTil_orig_categ', 'LotConfig_FR2_orig_categ', 'Fireplaces_cont_trans_normal', 'MSSubClass_120_orig_categ', '1stFlrSF_cont_trans_root_divide_HeatingQC_ord_trans_quadratic_trans_divide', 'CentralAir_ord_trans_normal_multiply_HalfBath_cont_trans_normal_trans_multiply', 'PoolQC_ord_trans_quadratic', 'TotalBsmtSF_cont_trans_root_divide_HeatingQC_ord_trans_quadratic_trans_divide', 'BldgType_Twnhs_orig_categ', 'RemodelAge_cont_trans_log', 'Foundation_PConc_orig_categ', 'MasVnrArea_cont_trans_normal', 'GarageType_BuiltIn_orig_categ', 'ScreenPorch_cont_trans_normal', 'SaleCondition_Abnorml_categ_multiply_GarageFinish_ord_trans_quadratic_trans_multiply', 'PavedDrive_ord_trans_quadratic', 'OverallCond_cont_trans_normal_multiply_HouseAge_cont_trans_root_trans_multiply', 'BsmtUnfSF_cont_trans_normal', 'BsmtFinType1_ord_trans_quadratic', 'HouseStyle_1.5Fin_orig_categ', 'LotShape_IR1_orig_categ', 'YrSold_2009_orig_categ', 'BedroomAbvGr_cont_trans_normal', 'GrLivArea_cont_trans_root_divide_BsmtFinType2_ord_trans_quadratic_trans_divide', 'YrSold_2010_orig_categ', 'Exterior2nd_Wd Sdng_orig_categ', 'RoofStyle_Hip_orig_categ', 'GarageIdx_cont_multiply_OverallCond_cont_trans_multiply', 'BsmtExposure_ord_trans_quadratic', 'LotFrontage_cont_multiply_SaleCondition_Abnorml_categ_trans_multiply', 'OverallCond_cont_trans_quadratic_multiply_Condition1_Norm_categ_trans_multiply', 'LowQualFinSF_cont_trans_root', 'BsmtFullBath_cont_trans_normal_multiply_BsmtExposure_ord_trans_quadratic_trans_multiply', 'FullBath_cont_trans_quadratic_multiply_KitchenQual_ord_trans_quadratic_trans_multiply', 'OverallCond_cont_trans_quadratic_multiply_OverallCond_cont_trans_normal_trans_multiply', 'BsmtUnfSF_cont_trans_root_multiply_OverallCond_cont_trans_normal_trans_multiply', 'OverallQual_cont_trans_quadratic_multiply_GarageQual_ord_trans_quadratic_trans_multiply', 'Functional_ord_trans_quadratic_multiply_OverallCond_cont_trans_multiply', '1stFlrSF_cont_trans_root_multiply_GarageIdx_cont_trans_multiply', 'Functional_ord_trans_quadratic_multiply_SummerSale_categ_trans_multiply', 'OverallQual_cont_trans_quadratic_multiply_KitchenAbvGr_cont_trans_normal_trans_multiply', 'OverallQual_cont_trans_quadratic_multiply_BsmtFullBath_cont_trans_normal_trans_multiply', 'GarageCars_cont_trans_quadratic_multiply_TotalBsmtSF_cont_trans_root_trans_multiply', 'OverallQual_cont_trans_quadratic_multiply_GarageIdx_cont_trans_multiply', 'MasVnrArea_cont_trans_root_multiply_GrLivArea_cont_trans_root_trans_multiply', 'Exterior1st_BrkFace_categ_multiply_SQFperRoom_cont_trans_multiply', 'MasVnrArea_cont_trans_root_multiply_Functional_ord_trans_quadratic_trans_multiply', 'GrLivArea_cont_multiply_neighborhood_index_cont_trans_normal_trans_multiply', 'neighborhood_index_cont_trans_normal_multiply_MasVnrArea_cont_trans_multiply', 'neighborhood_index_cont_trans_normal_multiply_TotalBsmtSF_cont_trans_multiply', 'GarageArea_cont_trans_root_multiply_OverallCond_cont_trans_normal_trans_multiply', 'TotRmsAbvGrd_cont_trans_quadratic_multiply_GarageIdx_cont_trans_quadratic_trans_multiply', 'GrLivArea_cont_multiply_GarageIdx_cont_trans_multiply', 'OverallCond_cont_trans_quadratic_multiply_LotArea_cont_trans_root_trans_multiply', '1stFlrSF_cont_trans_quadratic_divide_TotalSQF_cont_trans_root_trans_divide', 'SQFperRoom_cont_trans_quadratic', 'TotalSQF_cont_multiply_MasVnrType_ord_trans_quadratic_trans_multiply', 'EnclosedPorch_cont_trans_quadratic', 'TotalSQF_cont_trans_quadratic_divide_LotArea_cont_trans_root_trans_divide', 'SQFperRoom_cont_trans_quadratic_multiply_GarageQual_ord_trans_quadratic_trans_multiply', '2ndFlrSF_cont_trans_root_multiply_HouseAge_cont_trans_quadratic_trans_multiply', 'BsmtFinSF1_cont_trans_quadratic_divide_BsmtFinType1_ord_trans_quadratic_trans_divide', 'Exterior1st_BrkFace_categ_multiply_GrLivArea_cont_trans_quadratic_trans_multiply', 'TotalBsmtSF_cont_trans_quadratic', 'BsmtFinSF1_cont_trans_quadratic_multiply_Exterior1st_BrkFace_categ_trans_multiply', 'GrLivArea_cont_trans_quadratic_divide_BsmtFinType1_ord_trans_quadratic_trans_divide', 'GarageArea_cont_trans_quadratic_multiply_PrchSQ_cont_trans_root_trans_multiply', 'TotalSQF_cont_trans_quadratic_divide_CentralAir_ord_trans_normal_trans_divide', 'OpenPorchSF_cont_trans_root_multiply_BsmtUnfSF_cont_trans_quadratic_trans_multiply', 'GarageIdx_cont_multiply_GrLivArea_cont_trans_quadratic_trans_multiply', '2ndFlrSF_cont_trans_quadratic_multiply_1stFlrSF_cont_trans_root_trans_multiply', 'GarageIdx_cont_multiply_LotArea_cont_trans_quadratic_trans_multiply', 'Id']
    main(train_path, test_path, 0, output_path=r'../output/output.csv', keep_variables=keep_variables)

