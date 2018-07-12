"""Tests the import function defined in import_data.py
"""

from housing_prices import prep_features

def test_import():
    train_path = r'../../data/train.csv'
    new_df = housing_prices.import_data.process_data(train_path)
    assert new_df.shape == (1460, 81)

if __name__=='__main__':
    test_import()

