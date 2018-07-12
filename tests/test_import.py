"""Tests the import function defined in import_data.py
"""

import housing_prices

def test_import():
    train_path = r'./test_data/train.csv'
    new_df = housing_prices.import_data.process_data(train_path)
    assert new_df.shape == (1460, 81)

if __name__=='__main__':
    test_import()

