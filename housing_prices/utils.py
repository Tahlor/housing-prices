
def invert_dictionary(d):
    """Swap keys and values in dictionary, where values are a list. Lists should be mutually exclusive.

        Args:
            d (dict): Original dictionary, with key:[list] pairs
        Returns:
            dict: An inverted dictionary, where each element of list is a key, and the former key is the value
        """
    return dict( (v,k) for k in d for v in d[k] )


def reset_seed(seed=42):
    """Set all possible random seeds to reproduce results.

        Args:
            seed (int): random seed
        Returns:
            None
        """
    import numpy as np
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)

def checkEqual(L1, L2):
    """Compare lists for equal elements (ignore order).

        Args:
            L1 (list): first list
            L2 (list): second list
        Returns:
            bool: whether lists are equal
        """
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)


def drop_features(features, ignore_features):
    """Try to drop features that should be excluded from model, will not throw an error if variable not found

    Args:
        features (DataFrame): A DataFrame with features.
        ignore_features (list): A list (str) with features to exclude from DataFrame

    Returns:
        DataFrame: DataFrame without features in ignore_features list
    """
    for f in ignore_features:
        if f in features.columns:
            #features.drop(f, inplace=True, axis=1)
            features=features.drop(f, axis=1)
        else:
            print("{} not found in features".format(f))
    return features
