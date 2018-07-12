import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

op_dict = {"+":np.add, "*":np.multiply,"/":np.divide,"-":np.subtract, "add":np.add, "multiply":np.multiply,"divide":np.divide,"subtract":np.subtract}


def transform(features, rename=True, replace=False, trans_type=None, scaler=None, symmetric=None, verbose=0):
    """Transform a single variable or combine variables.

    Args:
        features (DataFrame): The dataframe containing data to rescale.
        rename (bool, optional): Rename variable using transformation descriptive variable name
        replace (bool, optional): Replace the original variable
        trans_type (str, optional): Type of transformation
        scaler (sklearn scaler object, optional): Scaler to use for transformation
        symmetric (bool, optional): True implies the operation to combine variables is commutative
        verbose (int, optional): Verbose printing

    Returns:
        tuple:
            DataFrame: All features, with newly transformed features
            Scaler obj: The scaler object used for the transform
    Except:
        AssertionError: If scaling type and scaler are not consistent.
    """

    vprint = print if verbose else lambda *a, **k: None

    if not rename and not replace:
        warnings.warn('Must rename if not replacing.')
        rename = True

    # If Scaler and trans_type are specified, verify they are consistent
    if not trans_type is None and not scaler is None:
        try:
            if "MinMaxScaler" in scaler.__repr__():
                assert trans_type == "scale01"
            elif "Standard" in scaler.__repr__():
                assert trans_type == "normal"
        except:
            raise AssertionError('Scaler and trans_type are not consistent')

    # Log, root, and inverse transforms require non-negative values
    if trans_type=="log":
        trans_features = np.log1p(features)
    elif trans_type=="inverse":
        trans_features = np.exp(-np.log1p(features))
    elif trans_type=="quadratic":
        trans_features = features**2
    elif trans_type=="cubic":
        trans_features = features**3
    elif trans_type=="root":
        trans_features = np.sqrt(features)
    elif trans_type == "normal":
        trans_features, scaler = rescale(features, scaling_type=trans_type)
    elif trans_type=="scale01":
        trans_features, scaler = rescale(features, scaling_type=trans_type)
    elif trans_type is None:
        if scaler is None:
            raise ValueError('Scaler and trans_type cannot both be None.')
        else:
            trans_features, _ = rescale(features, scaler=scaler)
            if "MinMaxScaler" in scaler.__repr__():
                trans_type = 'scale01'
            elif "Standard" in scaler.__repr__():
                trans_type = 'normal'
            else:
                raise ValueError('Unrecognized scaler type.')

    elif trans_type in ["add", "subtract", 'multiply', 'divide']:
        trans_features = perform_operations(features, trans_type, symmetric=symmetric, verbose=verbose)
    else:
        raise ValueError('Unrecognized transformation type.')

    if rename:
        new_cols = [str(col).replace("_orig","") + '_trans_' + trans_type for col in trans_features.columns]
        trans_features.columns = new_cols
    if replace:
        out = trans_features
    else: # if not replacing, append together, rename will have been called
        out = pd.concat([features, trans_features], axis=1)

    return out, scaler

def perform_operations(df, op, symmetric=None, verbose=0):
    """Perform an operation on all 2-variable combinations to create a new feature.

    Args:
        df (DataFrame): The dataframe containing data to rescale.
        op (str): String representing the op to be executed
        symmetric (bool, optional): True implies the operation to combine variables is commutative
        verbose (int, optional): Verbose printing

    Returns:
        DataFrame: Newly created features
    """

    vprint = print if verbose else lambda *a, **k: None
    _op = op_dict[op]

    if symmetric is None:
        symmetric = True if _op in [np.add, np.subtract, np.multiply] else False

    new_cols = {}

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
        warnings.warn("Duplicated columns detected, ignoring duplicates")

    ## Loop
    for idx, el in enumerate(df.columns):
        for idx2, el2 in enumerate(df.columns):
            if (idx>=idx2):
                continue

            df_el = df[el]
            df_el2 = df[el2]

            # Create new column
            col_name = el +"_"+ op +"_"+ el2
            new_col = perform_single_op(df_el, df_el2, op, col_name, verbose=verbose)
            if not new_col is None:
                new_cols[col_name] = new_col

            # Create reverse column
            if not symmetric:
                col_name = el2 + "_" + op + "_" + el
                new_col = perform_single_op(df_el2, df_el, op, col_name, verbose=verbose)
                if not new_col is None:
                    new_cols[col_name] = new_col

    # Concatenate
    vprint("Created {} variables for {}".format(len(new_cols),op))
    return_df = pd.DataFrame(new_cols)
    return return_df

def perform_single_op(df1, df2, op, col_name, verbose = 0):
    """Perform a single op on 2 columns to create a new feature.

    Args:
        df1 (DataFrame): The first variable/feature.
        df2 (DataFrame): The second variable/feature.
        op (str): String representing the op to be executed
        col_name (str): Name of new column
        verbose (int, optional): Verbose printing

    Returns:
        DataFrame: Newly created feature
    """

    vprint = print if verbose else lambda *a, **k: None
    _op = op_dict[op]

    # Don't perform division by zero
    if (df2 == 0).any() and op == "divide":
        vprint("{} involves division by zero, skipping".format(col_name))
        return None

    new_col = _op(df1, df2)

    if new_col.isna().values.any() or np.isinf(new_col).any():
        vprint("{} has inf/NaN result".format(col_name))
        return None

    vprint("Created {}".format(col_name))

    return new_col


def rescale(features, scaler = None, scaling_type= None, verbose=0):
    """Rescale a variable (e.g. to 0-1, -1-1 normal etc.

    Args:
        features (DataFrame): The dataframe containing data to rescale.
        scaler (sklearn scaler): If predefined scaler object available
        scalting_type (str): Type of scaler (str) if scaler not specified
        verbose (int, optional): Verbose printing

    Returns:
        tuple:
            DataFrame: All features, with newly transformed features
            Scaler obj: The scaler object used for the transform

    Except:
        ValueError: If scaling type and scaler are None, invalid etc.
    """
    vprint = print if verbose else lambda *a, **k: None

    if scaling_type is None and scaler is None:
        raise ValueError('Scaler and scaling_type cannot both be None.')
    if not (scaling_type is None and scaler is None):
        vprint("Scaler object and scaling type both specified, using scaler object")

    if scaler is None:
        if scaling_type =="normal":
            scaling_function = StandardScaler()
        elif scaling_type == "scale01":
            scaling_function = MinMaxScaler()
        else:
            raise ValueError('Unknown scaling_type value, consider "normal" or "scale01".')
        scaler = scaling_function.fit(features)
    scaled_features = pd.DataFrame(scaler.transform(features), index=features.index.values, columns=features.columns.values)
    return scaled_features, scaler
