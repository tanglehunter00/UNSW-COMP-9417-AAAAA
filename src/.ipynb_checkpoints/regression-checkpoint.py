import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import lightgbm as lgb

# ------------- XGBoost 可选 -------------
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ------------- PyTorch 可选 -------------
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


TARGETS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
HORIZONS = [1, 6, 12, 24]



# 工具函数

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def plot_series(true, pred, title, save_path):
    true = np.asarray(true)
    pred = np.asarray(pred)
    n = min(len(true), len(pred))
    plt.figure(figsize=(9, 3))
    plt.plot(true[:n], label="True")
    plt.plot(pred[:n], label="Pred")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def make_supervised(train, test, target, horizon, feature_cols):
    """
    传统表格模型使用：
    y(t+h) 为标签，当前时刻特征为 X。
    """
    Xtr = train[feature_cols].copy()
    ytr = train[target].shift(-horizon).dropna()
    Xtr = Xtr.iloc[: len(ytr)]

    Xte = test[feature_cols].copy()
    yte = test[target].shift(-horizon).dropna()
    Xte = Xte.iloc[: len(yte)]

    if len(Xtr) == 0 or len(Xte) == 0:
        return None, None, None, None

    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns)
    return Xtr_s, ytr, Xte_s, yte


def make_sequence_data(train, test, target, horizon, feature_cols, lookback=48):
    """
    深度时序模型使用：
    输入为过去 lookback 个时间步的特征序列，标签为 y(t+h)。
    """
    max_h = horizon

    def build(df):
        df = df.reset_index(drop=True)
        F = df[feature_cols].values
        y_full = df[target].values
        n = len(df)
        X_list, y_list = [], []
        for i in range(lookback - 1, n - max_h):
            X_list.append(F[i - lookback + 1 : i + 1])
            y_list.append(y_full[i + horizon])
        if not X_list:
            return None, None
        X_arr = np.stack(X_list, axis=0)
        y_arr = np.array(y_list)
        return X_arr, y_arr

    Xtr, ytr = build(train)
    Xte, yte = build(test)
    if Xtr is None or Xte is None:
        return None, None, None, None

    B, L, F = Xtr.shape
    scaler = StandardScaler()
    Xtr_flat = scaler.fit_transform(Xtr.reshape(B, L * F))
    Xte_flat = scaler.transform(Xte.reshape(Xte.shape[0], L * F))
    Xtr = Xtr_flat.reshape(B, L, F)
    Xte = Xte_flat.reshape(Xte.shape[0], L, F)
    return Xtr, ytr, Xte, yte


# =========================================================
# 深度学习模型
# =========================================================
if HAS_TORCH:

    class AttnLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.attn = nn.Linear(hidden_dim, 1)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            w = self.attn(out)                   # (B, L, 1)
            a = torch.softmax(w, dim=1)          # attention 权重
            ctx = (a * out).sum(dim=1)           # (B, H)
            y = self.fc(ctx).squeeze(-1)
            return y


    class SimpleTCN(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = x.transpose(1, 2)        # (B, F, L)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = x.mean(dim=2)            # global pooling
            y = self.fc(x).squeeze(-1)
            return y


    class TransformerEncoderTS(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
            super().__init__()
            self.input_linear = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_linear(x)     # (B, L, d_model)
            out = self.encoder(x)        # (B, L, d_model)
            ctx = out.mean(dim=1)
            y = self.fc(ctx).squeeze(-1)
            return y


    class SimpleTFT(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, hidden_dim=64):
            super().__init__()
            self.var_proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                activation="relu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            v = self.var_proj(x)
            z = self.encoder(v)
            g = self.gate(z)
            s = g * z
            ctx = s.mean(dim=1)
            y = self.fc(ctx).squeeze(-1)
            return y


    class SimpleInformer(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
            super().__init__()
            self.input_linear = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_linear(x)
            L = x.size(1)
            if L > 16:
                idx = torch.randperm(L)[: max(8, L // 3)]
                idx, _ = torch.sort(idx)
                x = x[:, idx, :]
            out = self.encoder(x)
            ctx = out.mean(dim=1)
            y = self.fc(ctx).squeeze(-1)
            return y


    class SimpleAutoformer(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
            super().__init__()
            self.input_linear = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_linear(x)
            out = self.encoder(x)
            att = (out @ out.transpose(1, 2)).mean(dim=1)
            w = torch.softmax(att, dim=-1)
            ctx = (w.unsqueeze(-1) * out).sum(dim=1)
            y = self.fc(ctx).squeeze(-1)
            return y


    def train_torch_model(model, Xtr, ytr, Xte, device, epochs=10, batch_size=128, lr=1e-3):
        model.to(device)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32)
        ds = TensorDataset(Xtr_t, ytr_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)
            preds = model(Xte_t).cpu().numpy()
        return preds


# =========================================================
# IQR 去异常（用于整套 clean 版本）
# =========================================================
def clean_dataset_with_iqr(df, targets=TARGETS):
    df_clean = df.copy()
    
    for col in targets:
        if col not in df_clean.columns:
            continue
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= low) & (df_clean[col] <= high)]
        after = len(df_clean)
        print(f"[CLEAN] {col}: removed {before - after} rows")
    return df_clean

# 主函数：训练所有模型（单次，给定一份 train）

def run_regression_models(train, test, feature_cols, outdir):
    os.makedirs(outdir, exist_ok=True)

    horizons = HORIZONS
    targets = TARGETS
    lookback = 48

    device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device = {device}")

    all_metrics = []

    for target in targets:
        if target not in train.columns or target not in test.columns:
            print(f"[WARN] missing target {target}, skip")
            continue

        for h in horizons:
            Xtr_tab, ytr_tab, Xte_tab, yte_tab = make_supervised(train, test, target, h, feature_cols)
            if Xtr_tab is None:
                print(f"[WARN] empty supervised set for {target}+{h}h, skip")
                continue

            preds = {}
            base_y = yte_tab

            # 传统模型 
            lgbm = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            lgbm.fit(Xtr_tab, ytr_tab)
            preds["LightGBM"] = pd.Series(lgbm.predict(Xte_tab), index=base_y.index)

            rf = RandomForestRegressor(n_estimators=200, random_state=0)
            rf.fit(Xtr_tab, ytr_tab)
            preds["RF"] = pd.Series(rf.predict(Xte_tab), index=base_y.index)

            lr = LinearRegression()
            lr.fit(Xtr_tab, ytr_tab)
            preds["Linear"] = pd.Series(lr.predict(Xte_tab), index=base_y.index)

            if HAS_XGB:
                xg = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    tree_method="hist",
                )
                xg.fit(Xtr_tab, ytr_tab)
                preds["XGBoost"] = pd.Series(xg.predict(Xte_tab), index=base_y.index)

            # 深度模型 
            if HAS_TORCH:
                Xtr_seq, ytr_seq, Xte_seq, yte_seq = make_sequence_data(
                    train, test, target, h, feature_cols, lookback=lookback
                )
            else:
                Xtr_seq = ytr_seq = Xte_seq = yte_seq = None

            seq_valid = HAS_TORCH and Xtr_seq is not None and Xte_seq is not None

            if seq_valid:
                input_dim = Xtr_seq.shape[2]
                models = {
                    "AttnLSTM": AttnLSTM(input_dim=input_dim),
                    "TCN": SimpleTCN(input_dim=input_dim),
                    "Transformer": TransformerEncoderTS(input_dim=input_dim),
                    "TFT": SimpleTFT(input_dim=input_dim),
                    "Informer": SimpleInformer(input_dim=input_dim),
                    "Autoformer": SimpleAutoformer(input_dim=input_dim),
                }

                for name, model in models.items():
                    preds_arr = train_torch_model(model, Xtr_seq, ytr_seq, Xte_seq, device)
                    # 与 tabular 的 yte 对齐
                    min_len = min(len(preds_arr), len(base_y))
                    idx = base_y.index[-min_len:]
                    preds[name] = pd.Series(preds_arr[-min_len:], index=idx)


            scores = {}
            for m, p in preds.items():
                common_idx = base_y.index.intersection(p.index)
                if len(common_idx) == 0:
                    continue
                val = rmse(base_y.loc[common_idx], p.loc[common_idx])
                scores[m] = val
                all_metrics.append({
                    "target": target,
                    "horizon": h,
                    "model": m,
                    "rmse": val,
                })

            #  BlendMean 
            blend_candidates = [m for m in ["LightGBM", "RF", "Linear", "XGBoost"] if m in preds]
            if len(blend_candidates) >= 2:
                common_idx = base_y.index
                for m in blend_candidates:
                    common_idx = common_idx.intersection(preds[m].index)
                if len(common_idx) > 0:
                    stack_vals = np.stack([preds[m].loc[common_idx].values for m in blend_candidates], axis=0)
                    blend_vals = stack_vals.mean(axis=0)
                    preds["BlendMean"] = pd.Series(blend_vals, index=common_idx)
                    val = rmse(base_y.loc[common_idx], preds["BlendMean"].loc[common_idx])
                    scores["BlendMean"] = val
                    all_metrics.append({
                        "target": target,
                        "horizon": h,
                        "model": "BlendMean",
                        "rmse": val,
                    })

            if not scores:
                print(f"[WARN] no valid scores for {target}+{h}h")
                continue

            # 选择最佳模型并画图
            best_model = min(scores, key=scores.get)
            print(f"[REG] {target}+{h}h best = {best_model} RMSE={scores[best_model]:.4f}")

            best_pred = preds[best_model]
            common_idx = base_y.index.intersection(best_pred.index)
            if len(common_idx) > 0:
                plot_series(
                    base_y.loc[common_idx],
                    best_pred.loc[common_idx],
                    f"{target} +{h}h {best_model}",
                    os.path.join(outdir, f"{target}_{h}h_{best_model}.png"),
                )

            # 保存每个模型的预测 CSV 
            for m, p in preds.items():
                common_idx = base_y.index.intersection(p.index)
                if len(common_idx) == 0:
                    continue
                df_pred = pd.DataFrame({
                    "y_true": base_y.loc[common_idx].values,
                    "y_pred": p.loc[common_idx].values,
                })
                df_pred.to_csv(
                    os.path.join(outdir, f"{target}_{h}h_{m}_pred.csv"),
                    index=False,
                )

    # 保存所有 RMSE
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    print(f"[INFO] Saved metrics to {os.path.join(outdir, 'metrics.csv')}")



# 包含去异常版本：一次跑出 base + clean
def run_regression_models_with_cleaning(train, test, feature_cols, root_outdir):
    base_dir = os.path.join(root_outdir, "base")
    clean_dir = os.path.join(root_outdir, "clean")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    print("Running regression models on RAW data ...")
    run_regression_models(train, test, feature_cols, base_dir)

    print("Cleaning train data with IQR for all targets ...")
    if target == "NMHC(GT)" and horizon == 24:
        train_clean = train
    else:

        train_clean = clean_dataset_with_iqr(train)

    print("Running regression models on CLEANED data ...")
    run_regression_models(train_clean, test, feature_cols, clean_dir)

def run_multihorizon_transformer(train, test, feature_cols, outdir):
    """
    Multi-horizon Transformer:
    - lookback = 168 hours
    - outputs = [1h, 6h, 12h, 24h]
    Saves RMSE to mh_transformer_metrics.csv
    """
    if not HAS_TORCH:
        print("[WARN] PyTorch not available, skipping multi-horizon Transformer.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    horizons = HORIZONS_MH
    metrics = []

    for target in targets:
        Xtr, Ytr, Xte, Yte = make_multihorizon_sequence_data(
            train, test, target, feature_cols, horizons=horizons, lookback=168
        )
        if Xtr is None or Xte is None:
            print(f"[WARN] no sequence data for multi-horizon {target}, skipping")
            continue

        input_dim = Xtr.shape[2]
        num_h = len(horizons)

        print(f"[MH-TRANS] Training multi-horizon Transformer for {target} on device={device}")
        model = MultiHorizonTransformerTS(
            input_dim=input_dim,
            num_horizons=num_h,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
        )
        preds = train_multi_output_model(model, Xtr, Ytr, Xte, device,
                                         epochs=20, batch_size=128, lr=1e-3)
        # preds, Yte: (N, num_horizons)
        for j, h in enumerate(horizons):
            y_true = Yte[:, j]
            y_pred = preds[:, j]
            val = rmse(y_true, y_pred)
            metrics.append({
                "target": target,
                "horizon": h,
                "model": "MH_Transformer",
                "rmse": val,
            })
            print(f"[MH-TRANS] {target}+{h}h RMSE={val:.4f}")

            # optional plot
            plot_series(
                y_true,
                y_pred,
                f"{target} +{h}h MH_Transformer",
                f"{outdir}/{target}_{h}h_MH_Transformer.png",
            )

    if metrics:
        df = pd.DataFrame(metrics)
        df.to_csv(f"{outdir}/mh_transformer_metrics.csv", index=False)
        print(f"[INFO] Saved multi-horizon Transformer metrics to {outdir}/mh_transformer_metrics.csv")
    else:
        print("[WARN] no metrics for multi-horizon Transformer.")
