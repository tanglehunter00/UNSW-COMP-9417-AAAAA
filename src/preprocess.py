import pandas as pd
import numpy as np
import os

def load_and_clean(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    
    df["Timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce"
    )
    
    # drop row without timestamp
    df = df.dropna(subset=["Timestamp"])
    
    # replace missing number with NaN
    df = df.replace({-200: np.nan})

    return df


def impute_base(df, cols):
    df = df.copy()
    df[cols] = df[cols].ffill()
    for c in cols:
        weekday_median = df.groupby(df["Timestamp"].dt.weekday)[c].transform("median")
        df[c] = df[c].fillna(weekday_median)
    df[cols] = df[cols].fillna(df[cols].median())
    return df

def add_time_features(df):
    ts = df["Timestamp"]
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df

def add_lag_ma_features(df, base_cols, lags, mas):
    for c in base_cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
        for w in mas:
            df[f"{c}_ma{w}"] = df[c].rolling(w, min_periods=max(1, w//2)).mean()
    return df

def temporal_split(df, train_end, test_end):
    train = df[df["Timestamp"] <= train_end].copy()
    test = df[(df["Timestamp"] > train_end) & (df["Timestamp"] <= test_end)].copy()
    return train, test

def build_feature_list(df):
    feature_cols = []
    for col in df.columns:
        if (
            "_lag" in col
            or "_ma" in col
            or col in ["hour", "weekday", "month", "is_weekend"]
            or col in ["T", "RH", "AH", "CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
        ):
            feature_cols.append(col)
    return feature_cols
