from transforms import transform
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import export_data
#from sklearn.impute import ChainedImputer
from fancyimpute import MICE
from sklearn.preprocessing import Imputer
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2

from scipy.stats import skew
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
TAGS = {}
# Impute LotFrontage
# neighborhood index
# season

#pd.set_option('display.height', 1000)
#pd.set_option('display.width', 1000)

def main(training_df, test_df, drop_outliers = True, use_log=False):
    global USE_LOG
    USE_LOG = use_log
    training_df, market_index, neighborhood_index = create_new_features(training_df)
    test_df, _, _ = create_new_features(test_df, market_index, neighborhood_index)

    training_df, targets = select_features(training_df, drop_outliers=drop_outliers, duplicate_outliers=False)
    test_df, test_targets = select_features(test_df, drop_outliers=False)
    training_df, test_df = delete_vacuous_variables(training_df, test_df)
    training_df, scaler_normal, scaler_01  = preprocess_features(training_df)
    test_df, _, _ = preprocess_features(test_df, scaler_normal=scaler_normal, scaler_01=scaler_01)

    #feature_reduction(training_df.drop(["SalePrice"], axis=1), training_df["SalePrice"])
    #feature_reduction(training_df, targets)

    #print(get_features(training_df, "index"))
    export_data.export_data(pd.concat([training_df, targets], axis=1),  "../output/processed_features.csv", True)
    export_data.export_data(pd.concat([test_df, test_targets], axis=1), "../output/processed_features_test.csv", True)

    return training_df, targets, test_df, test_targets


def create_market_index(features, index_name = "time", vars=["Season", "YrSold"]):
    index_name = index_name + "_index"
    temp = features[['SalePrice', 'TotalSQF'] + vars][:]
    temp[index_name]= (temp['SalePrice'] / temp['TotalSQF'])
    index = temp[[index_name]+vars].groupby(vars).mean()

    # Center at 1
    index[index_name] = index[index_name]/index[index_name].mean()
    print(index)
    return index

def create_new_features(features, market_index=None, neighborhood_index=None, MoSold_or_Season_index = "Season"):
    # Create new features
    features['TotalSQF'] = features['TotalBsmtSF'] + features['GrLivArea']
    features['NewHouse'] = np.less(abs(features['YrSold']-features['YearBuilt']),  2)
    features['HouseAge'] = features['YrSold'] - features['YearBuilt']
    features['SummerSale'] = np.isin(features["MoSold"], range(4,9))
    features['Season'] = ((features["MoSold"]/4)).astype(int)
    features['RemodelAge'] = features["YearRemodAdd"]-features["YrSold"]
    features['LargeHouse'] = features["TotalSQF"] > 4000

    features, tags = update_tags(features)

    vars = ["YrSold", MoSold_or_Season_index]
    if False:
        if market_index is None:
            market_index = create_market_index(features, "time", vars)
        features = pd.merge(features, market_index, how='left', left_on=vars,
                                right_on=vars)

    vars = ["Neighborhood"]
    if False:
        if neighborhood_index is None:
            neighborhood_index = create_market_index(features, "neighborhood", vars)
        features = pd.merge(features, neighborhood_index, how='left', left_on=vars,
                                right_on=vars)

    return features, market_index, neighborhood_index

def feature_reduction(features, targets, k=30):
    X_new = SelectKBest(chi2, k=k).fit_transform(features, targets)
    print(X_new)

def select_features(features, target_column = "SalePrice", features_to_remove=[], drop_outliers = True, duplicate_outliers = False, market_index=None):
    """Identify and prepare features.

    Args:
        url  (string): The download address.
        path (string): The location to save the download.

    Returns:
        bool: True if successful, False otherwise.
    """

    #features['neighborhood_index'] = features['neighborhood_index'] * features['TotalSQF']
    #features['time_index'] = features['time_index'] * features['TotalSQF']
    assert hasattr(features, 'tags')
    features, tags = update_tags(features)


    # Fix categorical variables expressed numerically
    features['MSSubClass'] = features['MSSubClass'].astype('category')
    features['YrSold'] = features['YrSold'].astype('category')
    features['Season'] = features['Season'].astype('category')
    #features['YearBuilt'] = features['YearBuilt'].astype('category')
    #features['YearRemodAdd'] = features['YearRemodAdd'].astype('category')

    features = recode(features)

    # Drop features
    features["HasPool"] = np.greater(features["PoolArea"], 0)
    features["PrchSQ"] = np.nansum(features[get_features(features,"porch")], axis=1)
    print(features["PrchSQ"])
    #features["PrchSQ"] = features["PrchSQ"].fillna(0)

    features = features.drop(["MoSold"], axis=1)
    features = features.drop(["Street"], axis=1)
    #features = features.drop(["GarageYrBlt"], axis=1)
    features = features.drop(["PoolQC"], axis=1)
    features = features.drop(["PoolArea"], axis=1)
    #features = features.drop(get_features(features, "Porch"), axis=1)
    #features = features.drop(("YrSold"), axis=1)
    features = features.drop(("BsmtFinSF1"), axis=1)
    features = features.drop(("BsmtFinSF2"), axis=1)
    features = features.drop(("MiscFeature"), axis=1)

    # Extra sq footage variables
    # Drop redundant variables
    features_to_remove += ["1stFlrSF", "TotRmsAbvGrd"]

    # Garage
    # keep GarageCars, GarageQual
    features = features.drop(["GarageFinish", "GarageArea", "GarageFinish", "GarageType"], axis=1)

    # Drop outliers
    if drop_outliers:
        # Drop cheap/expensive houses
        features = features[np.abs(features["SalePrice"]-features["SalePrice"].mean())<=(3.5*features["SalePrice"].std())]

        # Drop surprisingly priced houses
        price_per_sqf = (features['SalePrice'] / features['TotalSQF'])
        features = features[np.abs(price_per_sqf - price_per_sqf.mean()) <= (3.5 * price_per_sqf.std())]

    elif duplicate_outliers:
        price_per_sqf = (features['SalePrice'] / features['TotalSQF'])
        features = pd.concat([features[np.abs(price_per_sqf - price_per_sqf.mean()) > (3.5 * price_per_sqf.std())],features], axis=0)
        print(features.shape)

    # Keep normal sales
    if False:
        features = features.drop(np.isin(features["SaleType"],["WD", "New"]), axis=0)


    # Features with missigns
    #missing_features = features.columns[features.isnull().mean() > 0.8]
    #print(missing_features)
    #features = features.drop(missing_features, axis=1)

    export_data.export_data(features, "../output/features.csv", overwrite=True)

    # Choose y/target
    if target_column in features.columns:
        targets = features[[target_column]]
        targets["SalePriceMiscVal"] = targets[target_column] - features["MiscVal"]
        # Remove target column
        features = features.drop(target_column, axis=1)
        if USE_LOG:
            targets = np.log1p(targets)
            features["MiscVal"] = np.log1p(features["MiscVal"])
    else:
        targets = None

    # Remove any other features
    features = features.drop(features_to_remove, axis=1)

    # Keep only these features
    """features = features[['LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'ExterQual',
       'BsmtExposure', 'BsmtFinType1', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',
       'CentralAir', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       'FullBath', 'HalfBath', 'KitchenAbvGr', 'KitchenQual', 'Functional',
       'Fireplaces', 'FireplaceQu', 'GarageCars', 'PavedDrive', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Fence',
       'MiscVal', 'TotalSQF', 'HouseAge', 'SummerSale', 'RemodelAge', 'LotFrontage', 'Id']]
    """

    # Get dummy/one-hot encoded variables for categorical variables
    # Consider different approach here
    features.tags = tags
    new_features = pd.get_dummies(features)
    new_features.tags = tags

    # Update dummy variables
    for x in new_features.columns:
        if x not in features.columns:
            new_features.tags[x] = "categorical"

    return new_features, targets

def recode(features):
    assert hasattr(features, 'tags')

    # Re-encode ranked features
    utilities = {"AllPub":4, "NoSewr":3, "NoSeWa":2, "ELO":1, "NA":0}
    #qual = {"Ex":15, "Gd":4, "TA":3, "Fa":2, "Po":1, "NA":0}
    #qual = {"Ex": 8, "Gd": 6, "TA": 5, "Fa": 3, "Po": 1, "NA": 0}
    qual = {"Ex": 3, "Gd": 3, "TA": 2, "Fa": 2, "Po": 1, "NA": 0}

    basement_exp = {"Gd":10, "Av":6, "Mn":3, "No":2, "NA":0}
    basement_exp = {"Gd":4, "Av":3, "Mn":1, "No":1, "NA":0}

    #basement_fin = {"GLQ":6, "ALQ":5, "BLQ":4, "Rec":3, "LwQ":2, "Unf":1, "NA":0}
    basement_fin = {"GLQ":3, "ALQ":3, "BLQ":2, "Rec":2, "LwQ":1, "Unf":1, "NA":0}

    yes_no = {"Y":1, "N":0}

    #functional = {"Typ":20, "Min1":17, "Min2": 14, "Mod":10, "Maj1":8, "Maj2":6, "Sev":3, "Sal":0}
    #functional = {"Typ":6, "Min1":5, "Min2": 4, "Mod":3, "Maj1":2, "Maj2":1, "Sev":0, "Sal":0}
    functional = {"Typ": 3, "Min1": 3, "Min2": 3, "Mod": 2, "Maj1": 2, "Maj2": 1, "Sev": 0, "Sal": 0}

    garage_finish = {"Fin":8, "RFn":3, "Unf":2, 'NA':0}
    driveway = {"Y":2, "P":1, "N":0}
    fence = {"GdPrv":10, "MnPrv":8, "GdWo":7, "MnWw":4, "NA":0}
    mason = {"Stone":8,  "BrkFace":5, "BrkCmn": 2, "CBlock":0, "NA":0, "None":0}

    neighborhood = {"Blmngtn":105, "Blueste":99, "BrDale":102, "BrkSide":106, "ClearCr":103, "CollgCr":98, "Crawfor":106, "Edwards":98, "Gilbert":97, "IDOTRR":102, "MeadowV":90, "Mitchel":99, "NAmes":100, "NoRidge":101, "NPkVill":109, "NridgHt":104, "NWAmes":99, "OldTown":102, "SWISU":99, "Sawyer":101, "SawyerW":98, "Somerst":101, "StoneBr":104, "Timber":103, "Veenker":98}

    features = recode_features(features, "MasVnrType", mason)
    features = recode_features(features, "Utilities", utilities, missing=0) # NA = 0, just 3 of these
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
    features = recode_features(features, "FireplaceQu", qual, missing=0) # NA = 0
    features = recode_features(features, "GarageFinish", garage_finish)
    features = recode_features(features, "GarageQual", qual)
    features = recode_features(features, "GarageCond", qual)
    features = recode_features(features, "PavedDrive", driveway)
    features = recode_features(features, "PoolQC", qual, missing=0)
    features = recode_features(features, "Fence", fence, missing=0) # NA = 0
    #features = recode_features(features, "Neighborhood", neighborhood)
    return features

def smart_impute(features, features_to_impute = []):
    g = features.columns.to_series().groupby(features.dtypes).groups
    #print(g)
    if len(features_to_impute) == 0:
        features_to_impute = features.select_dtypes(include=['float64'])
    else:
        features_to_impute = features[features_to_impute]
    imputed_features = pd.DataFrame(MICE().complete(features_to_impute), index=features_to_impute.index.values, columns=features_to_impute.columns.values)
    return pd.concat([imputed_features, features.drop(features_to_impute, axis=1)], axis=1)

def chain_impute(features):
    imp = ChainedImputer(n_imputations=10, random_state=0)
    imp.fit(features)
    ChainedImputer(imputation_order='ascending', initial_strategy='mean',
                   max_value=None, min_value=None, missing_values=nan, n_burn_in=10,
                   n_imputations=10, n_nearest_features=None, predictor=None,
                   random_state=0, verbose=False)
    features = pd.DataFrame(imp.transform(features))
    return features


def get_features(features, search_term, ignore_case = True):
    if ignore_case:
        search_term = search_term.lower()
        return [x for x in features if search_term in x.lower()]
    else:
        return [x for x in features if search_term in x]

def recode_features(features, column, mapping, update_tag = 'ordinal', make_numeric=True, missing=None):
    # DF must have tags
    tags = features.tags
    features = features.replace({column: mapping})
    features.tags = tags

    if not update_tag is None:
        assert update_tag in ["categorical", "continuous", "ordinal"]
        features.tags[column]=update_tag
    if not missing is None:
        features[column] = features[column].fillna(missing)
    if make_numeric:
        features[column] = pd.to_numeric(features[column])

    print_null(features[[column]], print_entire_column=False)

    return features

def print_feature_dtypes(features):
    for f in features:
        print("{}: {}".format(f, features[f].dtype))

def print_null(df, print_entire_column=True):
    #print(df[[df.isna()]])
    numeric_cols = df.select_dtypes(include='number')

    na_cols = numeric_cols.columns[numeric_cols.isna().any()].tolist()
    if print_entire_column:
        print(numeric_cols[na_cols])
    else:
        print(numeric_cols[numeric_cols[na_cols].isna()])

def preprocess_features(features, scaler_normal = None, scaler_01 = None):
    """Prepare features, including handling missing values, normalization, PCA, etc.

    Args:
        url  (string): The download address.
        path (string): The location to save the download.

    Returns:
        bool: True if successful, False otherwise.
    """

    # Impute NaNs using the median
    #g = features.columns.to_series().groupby(features.dtypes).groups
    #print([x for x in g])
    # [dtype('float64'), dtype('uint8'), dtype('bool'), dtype('int64')]

    tags = features.tags
    #features = smart_impute(features, ["LotFrontage"])
    print(features["LotFrontage"])
    features = impute(features, strategy='median')
    print(features["LotFrontage"])

    # Split categorical features
    print_feature_dtypes(features)
    features.tags = tags
    id, continuous_features, categorical_features, ordinal_features = split_features(features, dont_rescale=["Id","MiscVal"])

    skewed_features = get_skewed_features(continuous_features)
    non_skewed_cont_features = [x for x in list(continuous_features) if x not in list(skewed_features)]

    #print(non_skewed_cont_features)
    continuous_features_log, _ = transform(continuous_features, rename=True, replace=True, trans_type="log")
    continuous_features_inverse, _ = transform(continuous_features, rename=True, replace=True, trans_type="inverse")
    continuous_features_normal, scaler_normal = transform(continuous_features, rename=True, replace=True, trans_type="normal", scaler=scaler_normal)
    continuous_features01, scaler_01 = transform(continuous_features, rename=True, replace=True, trans_type="scale01", scaler=scaler_01)
    continuous_features_root, _ = transform(continuous_features, rename=True, replace=True, trans_type="root")
    continuous_features_quad, _ = transform(continuous_features, rename=True, replace=True, trans_type="quadratic")
    df_list = [continuous_features_log, continuous_features_inverse, continuous_features_root, continuous_features, continuous_features_normal, continuous_features01, continuous_features_quad]
    #
    for df in df_list:
        if df.isnull().values.any():
            print(df.columns[0])
            print(df[df.isnull()])


    continuous_features = pd.concat(df_list, axis=1)

    """
    if USE_LOG:
        continuous_features[skewed_features] = np.log1p(continuous_features[skewed_features])
        continuous_features[non_skewed_cont_features], scaler = normalize(continuous_features[non_skewed_cont_features])
    else:
        # Normalize
        continuous_features, scaler = normalize(continuous_features, scaler)
    """



    # PCA
    #continuous_features = use_pca(continuous_features)

    # Recombine
    features = pd.concat([id, continuous_features, categorical_features, ordinal_features], axis=1)

    return features, scaler_normal, scaler_01

def get_skewed_features(features):
    skewed_feats = features.apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    return skewed_feats

def normalize(features, scaler = None):
    if scaler is None:
        #scaler = StandardScaler().fit(features)
        scaler = MinMaxScaler().fit(features)
    scaled_features = pd.DataFrame(scaler.transform(features), index=features.index.values, columns=features.columns.values)
    return scaled_features, scaler

def create_vacuous_variables(df1, df2, fill_value = 0):
    """Make sure training/test data have same features (take union of features)

    Args:
        url  (string): The download address.
        path (string): The location to save the download.

    Returns:
        bool: True if successful, False otherwise.
    """

    for c in df1.columns:
        if c not in df2.columns:
            df2[c] = fill_value
    for c in df2.columns:
        if c not in df1.columns:
            df1[c] = fill_value
    return df1, df2


def delete_vacuous_variables(df1, df2):
    """Make sure training/test data have same features (take intersection)

    Args:
        url  (string): The download address.
        path (string): The location to save the download.

    Returns:
        bool: True if successful, False otherwise.
    """

    tags1 = df1.tags
    tags2 = df2.tags

    df1 = df1.drop([c for c in df1 if c not in df2], axis=1)
    df2 = df2.drop([c for c in df2 if c not in df1], axis=1)

    df1.tags = tags1
    df2.tags = tags2
    return df1, df2


# Helper functions
def impute(dataframe, strategy="median"):
    if strategy=='0':
        imputed_DF = dataframe.fillna(0)
    else:
        fill_NaN = Imputer(missing_values=np.nan, strategy=strategy, axis=0)
        imputed_DF = pd.DataFrame(fill_NaN.fit_transform(dataframe))
        # imputed_DF = MAYBE COPY TAGS?
        imputed_DF.columns = dataframe.columns
        imputed_DF.index = dataframe.index
    return imputed_DF


# Split continuous and categorical features
def split_features(features, dont_rescale = ["Id"]):
    tags = features.tags
    dont_rescale = features[dont_rescale]
    features = features.drop(dont_rescale, axis=1)
    continuous_features = features[[x for x in tags if tags[x] == 'continuous'  and x in features.columns]]
    categorical_features = features[[x for x in tags if tags[x] == 'categorical' and x in features.columns]]
    ordinal_features = features[[x for x in tags if tags[x] == 'ordinal'  and x in features.columns]]
    return dont_rescale, continuous_features, categorical_features, ordinal_features

def use_pca(continuous_features, dim = 30):
    from sklearn.decomposition import PCA

    pca_scaled = PCA()
    pca_scaled.fit(continuous_features)
    cpts_scaled = pd.DataFrame(pca_scaled.transform(continuous_features))

    return cpts_scaled


def update_tags(df, tags=None, force_update=False):
    if tags is None and hasattr(df, 'tags'):
        tags = df.tags

    continuous_features = df.select_dtypes(include=['float', 'int']).columns.tolist()
    #ordinal_features = df.select_dtypes(include=['int']).columns.tolist()
    ordinal_features = []
    # If all values are 0,1, consider it categorical?
    categorical_features = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    assert len(df.columns) == len(continuous_features + ordinal_features + categorical_features)
    new_tags = invert({"categorical": categorical_features, "continuous": continuous_features, "ordinal": ordinal_features})

    # Update tags for columns that didn't have tags before
    if hasattr(df, 'tags') and not force_update:
        old_tags = df.tags.copy()
        for col in df.columns:
            if col not in old_tags:
                old_tags[col] = new_tags[col]
        new_tags = old_tags.copy()

    df.tags = new_tags
    return df, new_tags

def invert(d):
    return dict( (v,k) for k in d for v in d[k] )


if __name__=='__main__':
    pass


# Drop features with lots of missing
#### Normalize PCA features
# MCA
# PCA
# Don't use dummies - only use good variables - use them as a set?

# Regression, lasso etc.
# Remove stupid variables
# Year built...
# Ensemble

# Make feature processing class
# save out versions

# ENSEMBLE
# IMPUTE/missings better
# Ignore more variables
# Elastic net