"""Test the export function defined in export_data.py
"""
import os
import pandas as pd
from housing_prices import export_data

dirname = os.path.dirname(__file__)
export_path  = os.path.join(dirname, r'test_data/text-export.csv')

def test_export():

    if os.path.exists(export_path):
        os.remove(export_path)
        #print("Removing old file...")

    # Verify file does not exist
    assert not os.path.exists(export_path)

    # Create dataframe
    d = {'Id': [1, 2], 'col2': [3, 4]}
    original_df = pd.DataFrame(data=d)

    # Export dataframe
    export_data.export_data(original_df, export_path)

    # Verify file does exist
    assert os.path.exists(export_path)

    # Read in file
    df = pd.read_csv(export_path)

    # Assert all values are the same
    df.equals(original_df)

    # Remove file
    os.remove(export_path)

    # Verify file was deleted
    assert not os.path.exists(export_path)

if __name__=='__main__':
    test_export()

