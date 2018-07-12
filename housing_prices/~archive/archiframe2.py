import pandas as pd

class DataFrame(pd.DataFrame):
    @classmethod
    def convert_dataframe(cls, df):
        df.__class__ = cls
        return df

    def foo(self):
        self.m = ["this list"]

        return "Works"

    def guess_tags(self):
        # float64, bool, uint8, int64
        continuous_features = self.select_dtypes(include=['float'])
        ordinal_features = self.select_dtypes(include=['int'])
        # If all values are 0,1, consider it categorical?

        categorical_features = self.select_dtypes(include=['object', 'bool'])
        self.tags = {"categorical":categorical_features, "continuous":continuous_features, "ordinal":ordinal_features}



path = r'../data/train.csv'
df = pd.read_csv(path)
cdf = DataFrame.convert_dataframe(df)
print(cdf.foo())
print(cdf["SalePrice"])
cdf.guess_tags()