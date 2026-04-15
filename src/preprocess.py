import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def clean_data(df):
    df = df.copy()

    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    non_cat_cols = df.columns.difference(cat_cols)

    if len(non_cat_cols) > 0:
        df.loc[:, non_cat_cols] = df.loc[:, non_cat_cols].fillna(0)

    if len(cat_cols) > 0:
        df.loc[:, cat_cols] = df.loc[:, cat_cols].fillna("missing").astype(str)

    return df

def split_features_target(df):
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return X, y

def get_pipeline(X):
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    return preprocessor