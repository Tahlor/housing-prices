import pandas as pd

def process_data(path):
    """Takes a data file path input and returns a Pandas dataframe.

    Args:
        path (str): The path to the data file.

    Returns:
        DataFrame
    """
    df = pd.read_csv(path)
    return df

if __name__=="__main__":
    df = process_data(r'../data/train.csv')
