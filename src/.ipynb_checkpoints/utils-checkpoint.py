import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, confusion_matrix

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def discretize_CO(values):
    bins = [-np.inf, 1.5, 2.5, np.inf]
    labels = [0, 1, 2]
    return np.digitize(values, bins) - 1

def plot_series(y_true, y_pred, title, path):
    plt.figure(figsize=(10,3))
    n = min(500, len(y_true))
    plt.plot(range(n), y_true[:n], label="True")
    plt.plot(range(n), y_pred[:n], label="Pred")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_confusion(y_true, y_pred, h, fig_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    fig = plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"CO class +{h}h")
    plt.xticks([0,1,2], ["Low","Mid","High"])
    plt.yticks([0,1,2], ["Low","Mid","High"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"CO_cls_h{h}_cm.png"), dpi=160)
    plt.close(fig)

def load_optional_models():
    opt = {"reg": {}, "cls": {}}
    try:
        from xgboost import XGBRegressor, XGBClassifier
        from lightgbm import LGBMRegressor, LGBMClassifier
        from catboost import CatBoostRegressor, CatBoostClassifier
        opt["reg"]["XGB"] = XGBRegressor(n_estimators=400, learning_rate=0.05)
        opt["reg"]["LGBM"] = LGBMRegressor(n_estimators=400, learning_rate=0.05)
        opt["cls"]["XGBc"] = XGBClassifier(n_estimators=400, learning_rate=0.05)
        opt["cls"]["LGBMc"] = LGBMClassifier(n_estimators=400, learning_rate=0.05)
    except Exception:
        pass
    return opt
