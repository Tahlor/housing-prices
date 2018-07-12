import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

import export_data
from transforms import transform

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

def main(training_df, test_df, use_log=True, variable_combinations=[]):
    """Perform all data preprocessing steps, including:
        * Recoding variables, e.g. converting categorical to ordinal variables
        * Create novel features by combining old ones
        * Standardize features - Drop outliers, drop features, set feature types (e.g. categorical)
        * Impute missing variables
        * Create dummy/one-hot-encoded variables
        * "Rectangularize" data by ensuring train/test have the same features
        *  Scale, transform, or normalize features

    Args:
        training_df (DataFrame): The dataframe containing training data.
        test_df (DataFrame): The dataframe containing training data.

    Returns:
        (tuple): tuple containing:
            training_df(DataFrame): Processed features (X) for training data
            targets (DataFrame): Processed targets (y) for training data
            test_df(DataFrame): Processed features for test set
            test_targets (DataFrame): Processed targets for test set
    """

    ## RECODE VARIABLES
    training_df = recode(training_df)
    test_df = recode(test_df)
    print("Done recoding")

    ## CREATE NEW HUMAN-DERIVED FEATURES
    training_df, market_index, neighborhood_index = create_new_features(training_df)
    test_df, _, _ = create_new_features(test_df, market_index, neighborhood_index)
    print("Done creating new features")

    ## REMOVE FEATURES
    training_df, targets = feature_standardization(training_df, use_log=True)
    test_df, test_targets = feature_standardization(test_df, use_log=True)
    print("Done standardizing features")

    ## IMPUTE VARIABLES
    training_df = impute_missings(training_df)
    test_df = impute_missings(test_df)
    print("Done imputing variables")

    ## CREATE DUMMIES
    training_df = get_dummies(training_df)
    test_df = get_dummies(test_df)
    print("Done creating dummies")

    ## EXCLUDE DUMMY VARIABLES UNIQUE TO TEST SET
    test_df = create_vacuous_variables(reference=training_df, target=test_df, fill_value=0)
    print("Done excluding vacuous variables")

    ## TRANSFORM
    training_df, scaler_normal, scaler_01  = transform_features(training_df)
    test_df, _, _ = transform_features(test_df, scaler_normal=scaler_normal, scaler_01=scaler_01)

    ## Additional transforms
    training_df = additional_transforms(training_df, variable_combinations)
    test_df = additional_transforms(test_df, variable_combinations)

    ## Check for missing
    check_for_missing(training_df)
    check_for_missing(test_df)

    ## Export new features
    export_data.export_data(pd.concat([training_df, targets], axis=1),  "../output/processed_features.csv", True)
    export_data.export_data(pd.concat([test_df, test_targets], axis=1), "../output/processed_features_test.csv", True)

    return training_df, targets, test_df, test_targets


def additional_transforms(df, variable_combinations):
    """Take a list of variable names with defined transformations, and recreate these transforms.
        E.g.: A variable like var1_trans_quadratic_multiply_var2_trans_cubic_multiply would yield a new feature, defined by:
        var1**2 * var2**3

    Args:
        df (DataFrame): The dataframe containing features data.
        variable_combinations (list): List (str) of variables defined as above

    Returns:
        DataFrame: DataFrame with original + newly created features
    """

    for v in variable_combinations:
        if v not in df.columns:
            df = specific_transform(df, v)
    return df

def create_market_index(features, index_name = "time", vars=["Season", "YrSold"]):
    """Create e.g. temporal/geographical index variables, mapping average prices against another variable

    Args:
        features (DataFrame): The dataframe containing features data.
        index_name (str, optional): the name of the index
        vars (list, optional): List (str) of "key" variables to merge index on

    Returns:
        DataFrame: DataFrame with index values by "key" variables in vars list
    """

    index_name = index_name + "_index"
    temp = features[['SalePrice', 'TotalSQF'] + vars][:]
    temp[index_name]= (temp['SalePrice'] / temp['TotalSQF'])
    index = temp[[index_name]+vars].groupby(vars).mean()

    # Center at 1
    index[index_name] = index[index_name]/index[index_name].mean()
    #print(index)
    return index

def create_new_features(features, market_index=None, neighborhood_index=None, MoSold_or_Season_index = "Season"):
    """ Combine features to generate new features.

    Args:
        features (DataFrame): The dataframe containing training data.
        market_index (DataFrame): The dataframe mapping time periods to index values.
        neighborhood_index (DataFrame): The dataframe mapping neighborhoods to index values.
        MoSold_or_Season_index (str): Whether to index seasonly or monthly

    Returns:
        (tuple): tuple containing:
            features(DataFrame): DataFrame with original features combined with newly created ones
            market_index (DataFrame): The dataframe mapping time periods to index values.
            neighborhood_index (DataFrame): The dataframe mapping neighborhoods to index values.
    """

    # Create new features
    features['TotalSQF'] = features['TotalBsmtSF'] + features['GrLivArea']
    features['NewHouse'] = np.less(abs(features['YrSold']-features['YearBuilt']),  2)
    features['SummerSale'] = np.isin(features["MoSold"], range(4,9))
    features['Season'] = ((features["MoSold"]/4)).astype(int)
    features['RemodelAge'] = (features["YrSold"] - features["YearRemodAdd"]).clip(0,None)
    temp = (features["YrSold"] - features["GarageYrBlt"]).clip(0, None)
    features['GarageAgeInv'] = transform(temp, rename=False, replace=True, trans_type="inverse")[0]
    features['GarageIdx'] = features['GarageAgeInv'] * features['GarageFinish'] * features['GarageCars']
    features['HouseAge'] = (features['YrSold'] - features['YearBuilt']).clip(0,None)
    features['LargeHouse'] = features["TotalSQF"] > 4000
    features['SQFperRoom'] = features["TotalSQF"] / features.TotRmsAbvGrd
    features["HasPool"] = features["PoolArea"] > 0
    features["PrchSQ"] = np.nansum(features[get_features(features,"porch")], axis=1)

    merge_vars = ["YrSold", MoSold_or_Season_index]
    if market_index is None:
        market_index = create_market_index(features, "time", merge_vars)
    features = pd.merge(features, market_index, how='left', left_on=merge_vars,
                        right_on=merge_vars)

    merge_vars = ["Neighborhood"]
    if neighborhood_index is None:
        neighborhood_index = create_market_index(features, "neighborhood", merge_vars)
    features = pd.merge(features, neighborhood_index, how='left', left_on=merge_vars,
                        right_on=merge_vars)

    return features, market_index, neighborhood_index

def get_dummies(features):
    """ Get dummy/one-hot encoded variables for categorical variables

    Args:
        features (DataFrame): The dataframe containing training data.

    Returns:
        new_features(DataFrame): DataFrame with original features with categorical variables converted into binary dummy variables
    """

    new_features = pd.get_dummies(features)
    return new_features

def feature_standardization(features, target_column ="SalePrice", features_to_remove=[], use_log=True):
    """Prepare features, including dropping outliers, dropping features, setting feature types (e.g. categorical), selecting target/y feature, etc.

    Args:
        features (DataFrame): The dataframe containing training data.
        target_column (str): Name of target variable.
        features_to_remove(list(str)): Name of target variable.

    Returns:
        bool: True if successful, False otherwise.
    """

    # Fix categorical variables expressed numerically
    features['MSSubClass'] = features['MSSubClass'].astype('category')
    features['YrSold'] = features['YrSold'].astype('category')
    features['Season'] = features['Season'].astype('category')

    # Choose y/target
    if target_column in features.columns:
        targets = features[[target_column]]
        targets["SalePriceMiscVal"] = targets[target_column] - features["MiscVal"]
        # Remove target column
        features = features.drop(target_column, axis=1)
        if use_log:
            targets = np.log1p(targets)
            features["MiscVal"] = np.log1p(features["MiscVal"])
    else:
        targets = None

    # Remove any other features
    features = features.drop(features_to_remove, axis=1)
    return features, targets

def recode(features):
    """Recode variables, e.g. converting categorical to ordinal variables.

    Args:
        features (DataFrame): The dataframe containing data to recode.

    Returns:
        DataFrame: All features, with newly recoded features
    """

    # Re-encode ranked features
    utilities = {"AllPub":4, "NoSewr":3, "NoSeWa":2, "ELO":1, "NA":0}
    qual = {"Ex": 3, "Gd": 3, "TA": 2, "Fa": 2, "Po": 1, "NA": 0}
    basement_exp = {"Gd":4, "Av":3, "Mn":2, "No":1, "NA":0}
    basement_fin = {"GLQ":3, "ALQ":3, "BLQ":2, "Rec":2, "LwQ":1, "Unf":1, "NA":0}
    yes_no = {"Y":1, "N":0}
    functional = {"Typ":7, "Min1":6, "Min2": 5, "Mod":4, "Maj1":3, "Maj2":2, "Sev":1, "Sal":0}
    garage_finish = {"Fin":4, "RFn":3, "Unf":2, 'NA':0}
    driveway = {"Y":2, "P":1, "N":0}
    fence = {"GdPrv":5, "MnPrv":4, "GdWo":3, "MnWw":2, "NA":0}
    mason = {"Stone":8,  "BrkFace":5, "BrkCmn": 2, "CBlock":0, "NA":0, "None":0}
    neighborhood = {"Blmngtn":105, "Blueste":99, "BrDale":102, "BrkSide":106, "ClearCr":103, "CollgCr":98, "Crawfor":106, "Edwards":98, "Gilbert":97, "IDOTRR":102, "MeadowV":90, "Mitchel":99, "NAmes":100, "NoRidge":101, "NPkVill":109, "NridgHt":104, "NWAmes":99, "OldTown":102, "SWISU":99, "Sawyer":101, "SawyerW":98, "Somerst":101, "StoneBr":104, "Timber":103, "Veenker":98}

    features = recode_features(features, "ExterQual", qual)
    features = recode_features(features, "ExterCond", qual)
    features = recode_features(features, "BsmtQual", qual)
    features = recode_features(features, "BsmtCond", qual)
    features = recode_features(features, "BsmtExposure", basement_exp)
    features = recode_features(features, "BsmtFinType1", basement_fin)
    features = recode_features(features, "BsmtFinType2", basement_fin)
    features = recode_features(features, "HeatingQC", qual)
    features = recode_features(features, "CentralAir", yes_no)
    features = recode_features(features, "KitchenQual", qual)
    features = recode_features(features, "Functional", functional)
    features = recode_features(features, "GarageFinish", garage_finish)
    features = recode_features(features, "GarageQual", qual)
    features = recode_features(features, "GarageCond", qual)
    features = recode_features(features, "PavedDrive", driveway)

    # Where missing means NONE
    features = recode_features(features, "PoolQC", qual, missing=0)
    features = recode_features(features, "Fence", fence, missing=0) # NA = 0
    features = recode_features(features, "FireplaceQu", qual, missing=0) # NA = 0
    features = recode_features(features, "Utilities", utilities, missing=0) # NA = 0, just 3 of these
    features = recode_features(features, "MasVnrType", mason, missing=0)

    garage_features = get_features(features, "Garage")
    features[garage_features] = features[garage_features].fillna(0)

    return features

def get_features(features, search_term, ignore_case = True, inverse=False):
    """Find all variables that match a partial search string.

    Args:
        features (DataFrame): The dataframe containing data to search.
        search_term (str): The search term.
        ignore_case (bool): Whether to ignore case when finding matches.
        inverse (bool): Return all variables that don't match search string

    Returns:
        (list): list containing:
            (str) List of variables matching search string
    """

    if ignore_case:
        search_term = search_term.lower()
        return_list = [x for x in features if search_term in x.lower()]
    else:
        return_list =  [x for x in features if search_term in x]
    if inverse:
        return_list = [x for x in features.columns if x not in return_list]
    return return_list

def recode_features(features, column, mapping, make_numeric=True, missing=None):
    """Method that takes a dictionary mapping current feature values to desired values.

    Args:
        features (DataFrame): The dataframe containing data to search.
        column (str): The column to recode.
        mapping (dict): A dictionary of the form {current_variable_value:desired_variable_value}
        make_numeric (bool): Convert feature to numeric
        missing (value): A value to fill missing observations

    Returns:
        DataFrame: All features, with newly recoded features
    """

    features = features.replace({column: mapping})

    # Optionally fill missings
    if not missing is None:
        features[column] = features[column].fillna(missing)

    # Optionally convert to numeric
    if make_numeric:
        features[column] = pd.to_numeric(features[column])

    return features

def impute_missings(features):
    """Method that takes a dictionary mapping current feature values to desired values.

        Args:
            features (DataFrame): The dataframe containing data to search.
            column (str): The column to recode.
            mapping (dict): A dictionary of the form {current_variable_value:desired_variable_value}
            make_numeric (bool): Convert feature to numeric
            missing (value): A value to fill missing observations

        Returns:
            DataFrame: All features, with newly recoded features
        """

    features["LotFrontage"].fillna(value=0, inplace=True)
    features = impute(features, strategy='median')
    return features

def transform_features(features, scaler_normal = None, scaler_01 = None):
    """Prepare features, including handling missing values, normalization, PCA, etc.

    Args:
        url  (string): The download address.
        path (string): The location to save the download.

    Returns:
        bool: True if successful, False otherwise.
    """

    # Split categorical features

    tags = {'Condition1_RRAe_orig': 'categ', 'HouseStyle_SFoyer_orig': 'categ', 'MSSubClass_20_orig': 'categ',
     'RoofMatl_Tar&Grv_orig': 'categ', 'MSSubClass_45_orig': 'categ', 'MoSold_orig': 'cont',
     'HouseStyle_1.5Fin_orig': 'categ', 'Heating_GasW_orig': 'categ', 'Exterior1st_VinylSd_orig': 'categ',
     'Exterior1st_AsphShn_orig': 'categ', 'PavedDrive_orig': 'ord', 'LotShape_IR3_orig': 'categ',
     'Exterior1st_ImStucc_orig': 'categ', 'LotShape_IR1_orig': 'categ', 'MSSubClass_160_orig': 'categ',
     'SaleCondition_Partial_orig': 'categ', 'CentralAir_orig': 'ord', 'OpenPorchSF_orig': 'cont',
     'MSZoning_FV_orig': 'categ', 'BldgType_TwnhsE_orig': 'categ', 'SaleCondition_Alloca_orig': 'categ',
     'Exterior1st_BrkFace_orig': 'categ', 'LandContour_Lvl_orig': 'categ', 'SaleCondition_Normal_orig': 'categ',
     'GarageType_Attchd_orig': 'categ', 'BsmtFullBath_orig': 'cont', 'GarageIdx_orig': 'cont',
     'Exterior1st_Wd Sdng_orig': 'categ', 'SaleCondition_AdjLand_orig': 'categ', 'Exterior2nd_AsbShng_orig': 'categ',
     'Exterior2nd_Wd Shng_orig': 'categ', 'Exterior1st_MetalSd_orig': 'categ', 'Exterior2nd_CmentBd_orig': 'categ',
     'Neighborhood_NoRidge_orig': 'categ', 'PoolArea_orig': 'cont', '3SsnPorch_orig': 'cont',
     'RoofMatl_Metal_orig': 'categ', 'Neighborhood_Gilbert_orig': 'categ', 'Foundation_CBlock_orig': 'categ',
     'KitchenAbvGr_orig': 'cont', 'Street_Pave_orig': 'categ', 'RoofStyle_Gable_orig': 'categ',
     'HouseStyle_1Story_orig': 'categ', 'LotArea_orig': 'cont', 'Condition2_RRAe_orig': 'categ',
     'MiscFeature_Othr_orig': 'categ', 'Fireplaces_orig': 'cont', 'Exterior2nd_MetalSd_orig': 'categ',
     'Exterior2nd_HdBoard_orig': 'categ', 'SummerSale_orig': 'categ', 'SaleCondition_Abnorml_orig': 'categ',
     'Neighborhood_Crawfor_orig': 'categ', 'Neighborhood_CollgCr_orig': 'categ', 'Neighborhood_Veenker_orig': 'categ',
     'Condition1_Norm_orig': 'categ', 'GarageType_0_orig': 'categ', 'HouseStyle_SLvl_orig': 'categ',
     'Neighborhood_SawyerW_orig': 'categ', 'MSSubClass_85_orig': 'categ', 'OverallQual_orig': 'cont',
     'Exterior1st_Plywood_orig': 'categ', 'LotConfig_FR3_orig': 'categ', 'Heating_Wall_orig': 'categ',
     'Season_0_orig': 'categ', 'LandContour_Low_orig': 'categ', 'RemodelAge_orig': 'cont',
     'RoofStyle_Shed_orig': 'categ', 'MSSubClass_70_orig': 'categ', 'PoolQC_orig': 'ord', 'BsmtFinType1_orig': 'ord',
     'Exterior2nd_CBlock_orig': 'categ', 'MSZoning_RH_orig': 'categ', 'MSSubClass_75_orig': 'categ',
     'SQFperRoom_orig': 'cont', 'Neighborhood_Blmngtn_orig': 'categ', 'MSSubClass_120_orig': 'categ',
     'Neighborhood_StoneBr_orig': 'categ', 'MSSubClass_60_orig': 'categ', 'MiscFeature_Shed_orig': 'categ',
     'Exterior2nd_Wd Sdng_orig': 'categ', 'Foundation_Slab_orig': 'categ', 'Fence_orig': 'ord',
     'YrSold_2006_orig': 'categ', 'Condition2_PosA_orig': 'categ', 'OverallCond_orig': 'cont', 'BsmtCond_orig': 'ord',
     'BsmtExposure_orig': 'ord', 'Foundation_Stone_orig': 'categ', 'BedroomAbvGr_orig': 'cont',
     'LandContour_Bnk_orig': 'categ', 'MSSubClass_30_orig': 'categ', 'Foundation_Wood_orig': 'categ',
     'Exterior2nd_VinylSd_orig': 'categ', 'BsmtFinSF1_orig': 'cont', 'BldgType_Duplex_orig': 'categ',
     'MSSubClass_90_orig': 'categ', 'Neighborhood_MeadowV_orig': 'categ', 'FullBath_orig': 'cont',
     'BldgType_Twnhs_orig': 'categ', 'FireplaceQu_orig': 'ord', 'RoofStyle_Mansard_orig': 'categ',
     'Exterior1st_CBlock_orig': 'categ', 'Condition1_PosA_orig': 'categ', 'Season_3_orig': 'categ',
     'MSSubClass_80_orig': 'categ', 'ExterCond_orig': 'ord', 'GarageType_2Types_orig': 'categ',
     'LargeHouse_orig': 'categ', 'Exterior1st_CemntBd_orig': 'categ', 'HouseStyle_2.5Fin_orig': 'categ',
     'SaleType_WD_orig': 'categ', 'RoofMatl_CompShg_orig': 'categ', 'Exterior1st_HdBoard_orig': 'categ',
     'Electrical_SBrkr_orig': 'categ', 'RoofStyle_Flat_orig': 'categ', 'Foundation_PConc_orig': 'categ',
     'BsmtFinSF2_orig': 'cont', 'Neighborhood_IDOTRR_orig': 'categ', 'SaleType_ConLw_orig': 'categ',
     'LandSlope_Mod_orig': 'categ', 'Exterior1st_Stone_orig': 'categ', 'Exterior2nd_Stucco_orig': 'categ',
     'Heating_GasA_orig': 'categ', 'RoofMatl_WdShake_orig': 'categ', 'HouseAge_orig': 'cont',
     'Neighborhood_NPkVill_orig': 'categ', 'Utilities_orig': 'ord', 'Exterior2nd_AsphShn_orig': 'categ',
     'BsmtQual_orig': 'ord', 'GarageAgeInv_orig': 'cont', 'Exterior1st_BrkComm_orig': 'categ',
     'Electrical_Mix_orig': 'categ', 'Neighborhood_ClearCr_orig': 'categ', 'LotConfig_Corner_orig': 'categ',
     'SaleType_ConLD_orig': 'categ', 'BsmtHalfBath_orig': 'cont', 'YrSold_2010_orig': 'categ',
     'Electrical_FuseF_orig': 'categ', 'LotShape_Reg_orig': 'categ', 'MasVnrType_orig': 'ord',
     'Electrical_FuseP_orig': 'categ', 'Heating_Floor_orig': 'categ', 'GarageQual_orig': 'ord',
     'RoofStyle_Gambrel_orig': 'categ', 'Condition2_Norm_orig': 'categ', 'time_index_orig': 'cont',
     'GrLivArea_orig': 'cont', 'SaleType_Con_orig': 'categ', 'neighborhood_index_orig': 'cont',
     'GarageType_CarPort_orig': 'categ', 'Condition1_PosN_orig': 'categ', 'MiscVal_orig': 'cont',
     'Electrical_FuseA_orig': 'categ', 'Exterior1st_WdShing_orig': 'categ', 'BldgType_1Fam_orig': 'categ',
     'GarageCond_orig': 'ord', 'Neighborhood_BrkSide_orig': 'categ', 'Condition2_PosN_orig': 'categ',
     'LandContour_HLS_orig': 'categ', 'YrSold_2007_orig': 'categ', 'Neighborhood_BrDale_orig': 'categ',
     'MasVnrArea_orig': 'cont', 'SaleType_CWD_orig': 'categ', 'Heating_Grav_orig': 'categ', 'KitchenQual_orig': 'ord',
     'Neighborhood_NridgHt_orig': 'categ', 'LotConfig_Inside_orig': 'categ', 'RoofMatl_ClyTile_orig': 'categ',
     'WoodDeckSF_orig': 'cont', 'HeatingQC_orig': 'ord', 'Condition2_RRNn_orig': 'categ',
     'Neighborhood_Somerst_orig': 'categ', 'MSSubClass_40_orig': 'categ', 'MSZoning_C (all)_orig': 'categ',
     'ExterQual_orig': 'ord', 'MSSubClass_190_orig': 'categ', 'Exterior2nd_Stone_orig': 'categ',
     'Alley_Grvl_orig': 'categ', 'Neighborhood_Sawyer_orig': 'categ', 'Neighborhood_NWAmes_orig': 'categ',
     'LotFrontage_orig': 'cont', 'Exterior2nd_Brk Cmn_orig': 'categ', 'MSSubClass_180_orig': 'categ',
     'Season_2_orig': 'categ', 'Condition2_RRAn_orig': 'categ', 'BsmtFinType2_orig': 'ord',
     'Condition2_Artery_orig': 'categ', 'HasPool_orig': 'categ', 'GarageFinish_orig': 'ord',
     'SaleCondition_Family_orig': 'categ', 'EnclosedPorch_orig': 'cont', 'Foundation_BrkTil_orig': 'categ',
     'Condition1_RRAn_orig': 'categ', 'Exterior2nd_Other_orig': 'categ', 'HouseStyle_1.5Unf_orig': 'categ',
     'LotShape_IR2_orig': 'categ', 'HalfBath_orig': 'cont', 'Heating_OthW_orig': 'categ', 'LandSlope_Gtl_orig': 'categ',
     'TotRmsAbvGrd_orig': 'cont', 'Condition1_RRNe_orig': 'categ', 'MSZoning_RM_orig': 'categ',
     'Condition1_Feedr_orig': 'categ', 'GarageType_Detchd_orig': 'categ', 'TotalBsmtSF_orig': 'cont',
     'Exterior2nd_BrkFace_orig': 'categ', 'NewHouse_orig': 'categ', 'YrSold_2008_orig': 'categ',
     'RoofMatl_Roll_orig': 'categ', 'LotConfig_FR2_orig': 'categ', 'Neighborhood_Timber_orig': 'categ',
     'Neighborhood_Blueste_orig': 'categ', 'Condition2_Feedr_orig': 'categ', '2ndFlrSF_orig': 'cont',
     'LotConfig_CulDSac_orig': 'categ', 'Street_Grvl_orig': 'categ', 'Exterior1st_Stucco_orig': 'categ',
     'YrSold_2009_orig': 'categ', 'RoofStyle_Hip_orig': 'categ', 'BsmtUnfSF_orig': 'cont',
     'Neighborhood_NAmes_orig': 'categ', 'ScreenPorch_orig': 'cont', 'Functional_orig': 'ord',
     'GarageType_BuiltIn_orig': 'categ', 'Alley_Pave_orig': 'categ', 'Condition1_RRNn_orig': 'categ',
     'BldgType_2fmCon_orig': 'categ', 'LandSlope_Sev_orig': 'categ', 'Condition1_Artery_orig': 'categ',
     'Neighborhood_Edwards_orig': 'categ', 'GarageType_Basment_orig': 'categ', 'SaleType_New_orig': 'categ',
     'Season_1_orig': 'categ', 'MSSubClass_50_orig': 'categ', 'Neighborhood_SWISU_orig': 'categ',
     'Exterior2nd_ImStucc_orig': 'categ', 'HouseStyle_2.5Unf_orig': 'categ', 'HouseStyle_2Story_orig': 'categ',
     'RoofMatl_WdShngl_orig': 'categ', 'SaleType_COD_orig': 'categ', 'GarageArea_orig': 'cont',
     'MSZoning_RL_orig': 'categ', 'LowQualFinSF_orig': 'cont', 'Exterior1st_AsbShng_orig': 'categ',
     'Neighborhood_Mitchel_orig': 'categ', 'PrchSQ_orig': 'cont', 'Neighborhood_OldTown_orig': 'categ',
     'RoofMatl_Membran_orig': 'categ', 'MiscFeature_Gar2_orig': 'categ', '1stFlrSF_orig': 'cont',
     'Exterior2nd_Plywood_orig': 'categ', 'SaleType_ConLI_orig': 'categ', 'GarageCars_orig': 'cont',
     'TotalSQF_orig': 'cont', 'MiscFeature_TenC_orig': 'categ', 'SaleType_Oth_orig': 'categ'}

    # Add orig tag to columns
    features.columns = [str(col) + '_orig' if col != "Id" else "Id" for col in features.columns]

    # For now, treat ordinal and continuous variables the same
    dont_rescale = features[["Id", "MiscVal_orig"]]
    continuous_features = features[[feat for feat in tags.keys() if tags[feat] == "cont" and feat not in dont_rescale]]
    ordinal_features = features[[feat for feat in tags.keys() if tags[feat] == "ord" and feat not in dont_rescale]]
    categorical_features = features[[feat for feat in tags.keys() if tags[feat] == "categ" and feat not in dont_rescale]]

    # Add epithets
    continuous_features.columns = [str(col) + '_cont' for col in continuous_features.columns]
    ordinal_features.columns = [str(col) + '_ord' for col in ordinal_features.columns]
    categorical_features.columns = [str(col) + '_categ' for col in categorical_features.columns]


    continuous_features_log, _ = transform(continuous_features, rename=True, replace=True, trans_type="log")
    continuous_features_inverse, _ = transform(continuous_features, rename=True, replace=True, trans_type="inverse")
    continuous_features_normal, scaler_normal = transform(continuous_features, rename=True, replace=True, trans_type="normal", scaler=scaler_normal)
    continuous_features01, scaler_01 = transform(continuous_features, rename=True, replace=True, trans_type="scale01", scaler=scaler_01)
    continuous_features_root, _ = transform(continuous_features, rename=True, replace=True, trans_type="root")
    continuous_features_quad, _ = transform(continuous_features, rename=True, replace=True, trans_type="quadratic")
    df_list = [continuous_features_log, continuous_features_inverse, continuous_features_root, continuous_features, continuous_features_normal, continuous_features01, continuous_features_quad]

    continuous_features = pd.concat(df_list, axis=1)

    # Recombine
    features = pd.concat([dont_rescale, continuous_features, categorical_features, ordinal_features], axis=1)

    return features, scaler_normal, scaler_01

def check_for_missing(df):
    """Assert dataframe has no missing numeric values.

    Args:
        df (DataFrame): Dataframe to check

    Returns:
        None

    Exception:
        Assertion exception if dataframe has missing numeric values
    """

    numeric_df = df.select_dtypes(include='number')
    assert not (numeric_df.isna().values.any() or np.isinf(numeric_df.values).any())

def specific_transform(features, column_name, verbose=0):
    """Backout transformation from variable name. Pretty messy.

    Args:
        features (DataFrame): Dataframe with base features
        column_name (str): Name of column from which to derive transformation
        verbose (int, optional): Verbose messages for verbose>0

    Returns:
        DataFrame: DataFrame with only the newly created variable
    """

    vprint = print if verbose else lambda *a, **k: None

    # List all transform suffixes
    var_types = ["categ", "ord", "cont"]
    misc_keywords = ["trans"]
    comb_ops = ["add", "subtract", "multiply", "divide"]
    transforms = ["log", "inverse", "quadratic", "cubic", "root", "normal", "scale01"]
    all_keywords = misc_keywords + comb_ops + transforms

    c = column_name.replace(" ", "^%^")
    l = c.split("_")

    # Get the "base" variables
    base_variables = '_'.join([x if x not in all_keywords else ' ' for x in l])
    base_variables = re.sub("(_ )+_", " ", base_variables).strip().split(" ")
    base_variables = [b.replace("^%^", " ") for b in base_variables] # ugly hack for dealing with spaces
    vprint(base_variables)

    # Process "base" variable to get original, untransformed variable (e.g. add orig to it etc.)
    original_variables = base_variables[:]
    for n, o in enumerate(original_variables):
        for var_type in var_types:
            original_variables[n] = original_variables[n].replace(var_type, "orig_" + var_type)
    vprint(original_variables)

    # Get string defining transformations
    for o in set(base_variables):
        column_name = column_name.replace(o, "|" + o + "|")
    parsed_variables = column_name.split("|")[1:]

    vprint(parsed_variables)

    # Create a list containing [[original_var_name, str_of_transforms],...]
    var_list = []
    for n in range(0, len(parsed_variables), 2):
        var = original_variables[int(n/2)]
        trans = parsed_variables[n + 1]
        var_list.append([var, trans])

    # Check for individual transforms
    vprint(var_list)
    for var in var_list:
        feat_col = features[[var[0]]]
        for t in transforms:
            if t in var[1]:
                new_df = transform(feat_col, rename=True, replace=True, trans_type=t, verbose=0)[0]
                var.append(new_df)
                break  # Assume one transform
        if len(var)<3:
            var.append(feat_col)

    # Check for variable combinations
    if len(var_list) > 1:
        new_features = pd.concat([var_list[0][2], var_list[1][2]], axis=1)
        for op in comb_ops:
            if op in var_list[0][1]:
                combined_feature = transform(new_features, rename=True, replace=True, trans_type=op, verbose=0)[0]
                break
    else:
        combined_feature = var_list[0][2]
    combined_feature = pd.concat([features, combined_feature], axis=1)
    return combined_feature

def create_vacuous_variables(reference, target, fill_value = 0):
    """Make sure training/test data have same features (take union of features). This can arise from creating one-hot encoded variables if one DataFrame has feature values not observed in the other DataFrame.

    Args:
        reference (DataFrame): Dataframe with complete set of variables for model
        target (DataFrame): Dataframe with possibly incomplete set of variables

    Returns:
        DataFrame: The result DataFrame corresponding to the target DataFrame argument passed
    """


    for c in reference.columns:
        if c not in target.columns:
            target[c] = fill_value

    return target


# Helper functions
def impute(df, strategy="median"):
    """Impute numerical variables.

    Args:
        df (DataFrame): Dataframe with features to impute
        strategy (str): Imputation method (e.g. median, mean etc.)

    Returns:
        DataFrame: DataFrame with imptued values
    """

    numeric_cols = df.select_dtypes(include='number')
    other_cols = df.select_dtypes(exclude='number')

    if strategy=='0':
        imputed_DF = numeric_cols.fillna(0)
    else:
        fill_NaN = Imputer(missing_values=np.nan, strategy=strategy, axis=0)
        imputed_DF = pd.DataFrame(fill_NaN.fit_transform(numeric_cols))
        imputed_DF.columns = numeric_cols.columns
        imputed_DF.index = numeric_cols.index
    return pd.concat([imputed_DF,other_cols],1)
