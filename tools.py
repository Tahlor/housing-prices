import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)


def print_null(df, option='column_names', include_negative=False, include_inf=True):
    numeric_cols = df.select_dtypes(include='number')
    na_cols = numeric_cols.columns[numeric_cols.isna().any()].tolist()
    if include_inf:
        inf_cols = list(numeric_cols.columns.to_series()[np.isinf(numeric_cols).any()])
        na_cols = list(set(inf_cols+na_cols))
    if include_negative:
        neg_cols = numeric_cols.columns[(numeric_cols<0).any()].tolist()
        na_cols = list(set(neg_cols+na_cols))
    na_df = numeric_cols[na_cols]
    print("Null columns:")
    if option == 'column_names':
        print(na_cols)
    elif option == 'columns':
        print(numeric_cols[na_cols])
    elif option == 'only_null':
        if include_negative:
            print(na_df[na_df.isnull().any(axis=1)|(na_df<0).any(axis=1)][na_cols])
        else:
            print(numeric_cols[numeric_cols.isnull().any(axis=1)][na_cols])

def count_nulls(df):
    null_columns = df.columns[df.isnull().any()]
    print(df[null_columns].isnull().sum())


def scatter(df, var1, var2, joint=True):
    if not joint:
        plt.scatter(df[var1], df[var2])
        plt.title("{} vs {}".format(var1, var2))
        plt.ylabel(var2)
        plt.xlabel(var1)
    else:
        sns.jointplot(df[var1], df[var2], color='gold');


def corr_matrix(df, varlist= ["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]):
    corrMatrix = df[varlist].corr()
    sns.set(font_scale=1.10)
    sns.set(font_scale=1.10)
    plt.figure(figsize=(10, 10))
    sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
                square=True, annot=True, cmap='viridis', linecolor="white")
    plt.title('Correlation between features');

def distribution(df, variable):
    sns.distplot(df[variable], color="r")
    plt.title("Distribution of {}".format(variable))
    plt.ylabel("Number of Occurences")
    plt.xlabel(variable);

def missing_values(df):
    null_columns = df.columns[df.isnull().any()]
    df[null_columns].isnull().sum()
    labels = []
    values = []
    for col in null_columns:
        labels.append(col)
        values.append(df[col].isnull().sum())
    ind = np.arange(len(labels))
    width = 0.9
    fig, ax = plt.subplots(figsize=(12, 50))
    rects = ax.barh(ind, np.array(values), color='violet')
    ax.set_yticks(ind + ((width) / 2.))
    ax.set_yticklabels(labels, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_ylabel("Column Names")
    ax.set_title("Variables with missing values");


def worst_predictions(prediction, test, top=20):
    error = np.abs(test-prediction)


def print_feature_dtypes(features):
    for f in features:
        print("{}: {}".format(f, features[f].dtype))
