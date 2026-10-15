import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss",
            "hours-per-week", "net_capital", "has_capital", "is_overtime"]
CAT_COLS = ["workclass", "marital-status", "occupation", "relationship",
            "race", "sex", "native-country"]


def load_data():
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X = adult.data.drop(columns=["fnlwgt", "education"])
    y = (adult.target == ">50K").astype(int)
    return X, y


def split_data(X, y):
    # 60 / 20 / 20 stratifié
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def add_features(df):
    df = df.copy()
    df["net_capital"] = df["capital-gain"].astype(float) - df["capital-loss"].astype(float)
    df["has_capital"] = ((df["capital-gain"] > 0) | (df["capital-loss"] > 0)).astype(int)
    df["is_overtime"] = (df["hours-per-week"] > 40).astype(int)
    for c in CAT_COLS:  # les NaN restent des NaN
        df[c] = df[c].astype("object")
    return df


def get_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("columns", ColumnTransformer([
            ("num", numeric, NUM_COLS),
            ("cat", categorical, CAT_COLS),
        ])),
    ])


if __name__ == "__main__":
    X, y = load_data()
    print(X.shape)
    print(y.value_counts(normalize=True))
    print("Valeurs manquantes :\n", X.isna().sum()[X.isna().sum() > 0])
