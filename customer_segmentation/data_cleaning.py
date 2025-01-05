def summarize_nulls(df):
    print("Total NaN values per column:")
    print(df.isna().sum())

def summarize_negatives(df):
    print("Total negative values per column:")
    df_numerical = df.select_dtypes(include=["number"])
    print((df_numerical < 0).sum())

def summarize_zeros(df):
    print("Total zero values per column:")
    df_numerical = df.select_dtypes(include=['number'])
    print((df_numerical == 0).sum())

def filter_nulls(df, column=None):
    if column:
        return df[df[column].isnull()]
    else:
        return df[df.isnull().any(axis=1)]

def filter_negatives(df, column=None):
    if column:
        return df[df[column] < 0]
    else:
        df_numerical = df.select_dtypes(include=["number"])
        return df[(df_numerical < 0).any(axis=1)]

def filter_zeros(df, column=None):
    if column:
        return df[df[column] == 0]
    else:
        df_numerical = df.select_dtypes(include=["number"])
        return df[(df_numerical == 0).any(axis=1)]

def remove_nulls(df):
    return df.dropna()

def remove_negatives(df):
    df_numerical = df.select_dtypes(include=['number'])
    return df[(df_numerical >= 0).all(axis=1)]

def remove_zeros(df):
    df_numerical = df.select_dtypes(include=['number'])
    return df[~(df_numerical == 0).any(axis=1)]