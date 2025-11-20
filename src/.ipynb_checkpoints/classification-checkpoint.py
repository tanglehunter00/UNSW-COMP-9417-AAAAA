import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False


TARGET = "CO(GT)"
HORIZONS = [1, 6, 12, 24]


def discretize_CO_missing_aware(series):
    miss_mask = series.isna().astype(int)
    s_interp = series.interpolate("linear").bfill().ffill()
    bins = [0, 2, 4, 8, 50]
    labels = [0, 1, 2, 3]
    s_discrete = pd.cut(s_interp, bins=bins, labels=labels).astype(int)
    return s_discrete, miss_mask


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_classification_models(train, test, feature_cols, outdir):

    outdir = os.path.join(outdir, "classification")
    os.makedirs(outdir, exist_ok=True)

    results = []

    for h in HORIZONS:

        print(f"[CLASS] Processing CO classification h={h} ...")

        ytr_raw = train[TARGET].shift(-h)
        yte_raw = test[TARGET].shift(-h)

        ytr, miss_tr = discretize_CO_missing_aware(ytr_raw)
        yte, miss_te = discretize_CO_missing_aware(yte_raw)

        ytr = ytr.iloc[:-h]
        yte = yte.iloc[:-h]
        miss_tr = miss_tr.iloc[:-h]
        miss_te = miss_te.iloc[:-h]

        Xtr = train[feature_cols].iloc[:-h].copy()
        Xte = test[feature_cols].iloc[:-h].copy()

        Xtr["CO_missing_mask"] = miss_tr.values
        Xte["CO_missing_mask"] = miss_te.values

        if len(ytr) < 50 or len(yte) < 20:
            print(f"[WARN] horizon {h} has too few samples.")
            continue

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        models = {
            "Logistic": LogisticRegression(max_iter=300),
            "RF": RandomForestClassifier(n_estimators=200, random_state=0),
            "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=300),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03
            ),
        }

        if HAS_XGB:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss"
            )

        for name, model in models.items():
            print(f"[CLASS] Training {name} for h={h} ...")

            model.fit(Xtr_s, ytr)
            pred = model.predict(Xte_s)

            acc = accuracy_score(yte, pred)
            f1m = f1_score(yte, pred, average="macro")
            f1w = f1_score(yte, pred, average="weighted")

            print(f"[CLASS] {name} h={h}: Acc={acc:.4f}  F1m={f1m:.4f}")

            df_pred = pd.DataFrame({
                "y_true": yte.values,
                "y_pred": pred,
                "missing_mask": miss_te.values
            })
            df_pred.to_csv(
                os.path.join(outdir, f"CO_{h}h_{name}_pred.csv"),
                index=False
            )

            plot_confusion_matrix(
                yte,
                pred,
                classes=[0, 1, 2, 3],
                save_path=os.path.join(outdir, f"CO_{h}h_{name}_cm.png")
            )

            results.append({
                "horizon": h,
                "model": name,
                "acc": acc,
                "f1_macro": f1m,
                "f1_weighted": f1w
            })

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(outdir, "classification_metrics.csv"), index=False)

    print(f"[CLASS] All metrics saved → {outdir}/classification_metrics.csv")
