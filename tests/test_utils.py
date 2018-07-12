"""Tests the import function defined in utils.py
"""

import numpy as np
import pandas as pd
import utils

feats_with_id = pd.DataFrame({'col1': [3, 4], 'Id': [1, 2]})
feats = pd.DataFrame(np.asarray([3, 4]))
targs = pd.DataFrame(np.asarray([1, 6]))

def test_drop_features():
    assert ["Id"] in feats_with_id.columns.values
    return_df  = utils.drop_features(feats_with_id, ["Id"])
    assert not "Id" in return_df.columns.values

def test_invert_dictionary():
    d1 = {1:(2,5), 3:(10,11)}
    d2 = {2: 1, 5: 1, 10:3, 11:3}
    d2_new = utils.invert_dictionary(d1)
    assert d2 == d2_new

def test_checkEqual():
    x = [-5,7,-2]
    y = [4,7,-2]
    z = [-2,7,-5]
    assert utils.checkEqual(z,x)
    assert not utils.checkEqual(y, x)

def test_reset_seed():
    utils.reset_seed(47)
    assert np.random.get_state()[1][0] == 47

if __name__=='__main__':
    #test_drop_features()
    #test_invert_dictionary()
    #test_checkEqual()
    test_reset_seed()