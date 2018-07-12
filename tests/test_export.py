"""Test the export function defined in export_data.py
"""
import os
import pandas as pd
from housing_prices import export_data

dirname = os.path.dirname(__file__)
export_path = os.path.join(dirname, r'test_data/text-export.csv')
export_path1 = os.path.join(dirname, r'test_data/text-export v_01.csv')


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

    # Verify incrementer works
    export_data.export_data(original_df, export_path)

    # Verify files exist
    assert os.path.exists(export_path)
    assert os.path.exists(export_path1)

    # Read in file
    df = pd.read_csv(export_path)

    # Assert all values are the same
    df.equals(original_df)

    # Remove file
    os.remove(export_path)
    os.remove(export_path1)

    # Verify file was deleted
    assert not os.path.exists(export_path)
    assert not os.path.exists(export_path1)

def test_increment_path_version():
    path=export_data.increment_path_version("./this.txt")
    open(path, 'a').close()
    path2 = export_data.increment_path_version("./this.txt")

    assert not os.path.exists(path2)
    open(path2, 'a').close()
    assert path2!=path

    os.remove(path)
    os.remove(path2)



if __name__=='__main__':
    test_export()
    test_increment_path_version()
