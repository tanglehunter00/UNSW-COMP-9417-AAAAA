import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import os


TARGETS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
HORIZONS = [1, 6, 12, 24]


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))



def remove_outliers_iqr(df, col):
    """
    使用 IQR 四分位法去除异常点
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    cleaned = df[(df[col] >= low) & (df[col] <= high)]
    return cleaned



def make_supervised(df, target, horizon, feature_cols):
    df = df.copy()
    df["y"] = df[target].shift(-horizon)

    df = df.dropna(subset=["y"])
    if len(df) < 100:
        return None, None, None, None

    X = df[feature_cols]
    y = df["y"]

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    split = int(len(df) * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]



def run_anomaly_detection(train_df, feature_cols, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    metrics = []

    for target in TARGETS:
        for h in HORIZONS:

            print(f"Processing {target} +{h}h ...")


            Xtr, ytr, Xte, yte = make_supervised(train_df, target, h, feature_cols)

            if Xtr is None:
                print(f"[WARN] {target}+{h}h 数据太少，跳过")
                continue

    
            base_lgb = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8
            )
            base_lgb.fit(Xtr, ytr)
            base_pred = base_lgb.predict(Xte)
            base_rmse = rmse(yte, base_pred)


            cleaned_df = remove_outliers_iqr(train_df, target)
            Xtr2, ytr2, Xte2, yte2 = make_supervised(cleaned_df, target, h, feature_cols)

            if Xtr2 is None:
                print(f"[WARN] {target}+{h}h 清洗后数据太少，跳过 robust")
                continue

            robust_lgb = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8
            )
            robust_lgb.fit(Xtr2, ytr2)
            robust_pred = robust_lgb.predict(Xte2)
            robust_rmse = rmse(yte2, robust_pred)

  
            metrics.append({
                "target": target,
                "horizon": h,
                "model": "LightGBM",
                "rmse_base": base_rmse,
                "rmse_robust": robust_rmse
            })

            print(f"{target} +{h}h: base RMSE={base_rmse:.4f}, robust RMSE={robust_rmse:.4f}")

    # 保存 CSV
    df = pd.DataFrame(metrics)
    df.to_csv(f"{outdir}/anomaly_metrics.csv", index=False)
    print(f"[INFO] 已保存 anomaly_metrics.csv -> {outdir}/anomaly_metrics.csv")
