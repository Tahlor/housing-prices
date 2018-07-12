import pandas as pd

class X(pd.DataFrame):
    def __init__(self, df):
        df.__class__ = X
        self = df

    def test(self):
        print("TEST")
    # def guess_tags(self):
    #     # float64, bool, uint8, int64
    #     continuous_features = self.select_dtypes(include=['float'])
    #     ordinal_features = self.select_dtypes(include=['int'])
    #     # If all values are 0,1, consider it categorical?
    #
    #     categorical_features = self.select_dtypes(include=['object', 'bool'])
    #
    #     return {"categorical":categorical_features, "continuous":continuous_features, "ordinal":ordinal_features}

path = r'../data/train.csv'
x = pd.read_csv(path)
df = X(x)
print(x.SalePrice)
print(x.test())
print(df["SalePrice"])

if False:
    df = (pd.read_csv(path))
    df.this_new_attribute = 5
    print(df.this_new_attribute )
    #print(df.tags)

    # df.__class__= DataFrame
    # print(df["SalePrice"])
    # df.test()
