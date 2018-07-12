"""Tests the import function defined in import_data.py
"""

import housing_prices
import os
dirname = os.path.dirname(__file__)
train_path = os.path.join(dirname, r'test_data/train.csv')

def test_import():

    new_df = housing_prices.import_data.process_data(train_path)
    assert new_df.shape == (1460, 81)

if __name__=='__main__':
    test_import()

