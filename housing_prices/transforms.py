import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


def transform(features, rename=True, replace=False, trans_type=None, scaler=None, symmetric=None, verbose=0):
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
        trans_features = perform_operation(features, trans_type, symmetric=symmetric, verbose=verbose)
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

import numpy

op_dict = {"+":np.add, "*":np.multiply,"/":np.divide,"-":np.subtract, "add":np.add, "multiply":np.multiply,"divide":np.divide,"subtract":np.subtract}
def perform_operation(df, op, verbose=0, symmetric=None):
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

            col_name = el +"_"+ op +"_"+ el2
            new_col = _op(df_el, df_el2)

            # Skip null columns
            if new_col.isna().values.any() or np.isinf(new_col).any():
                continue
            new_cols[col_name] = new_col

            if not symmetric:
                new_col2 = _op(df_el2, df_el)
                col_name2 = el2 + "_" + op + "_" + el
                if new_col2.isna().values.any() or np.isinf(new_col2).any():
                    print("{} has inf/NaN result".format(col_name2))
                    continue
                new_cols[col_name2] = new_col2

            if verbose:
                vprint("Created {}".format(col_name))

    # Concatenate
    vprint("Created {} variables for {}".format(len(new_cols),op))
    return_df = pd.DataFrame(new_cols)
    return return_df

def rescale(features, scaler = None, scaling_type= None):
    if scaling_type is None and scaler is None:
        raise ValueError('Scaler and scaling_type cannot both be None.')
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


