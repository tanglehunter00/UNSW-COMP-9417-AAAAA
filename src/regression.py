import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import lightgbm as lgb

try:
    import xgboost as xgb
    has_xgboost = True
except ImportError:
    has_xgboost = False
    print("xgboost not available")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    has_pytorch = True
except ImportError:
    has_pytorch = False
    print("pytorch not available, deep models will be skipped")

target_list = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
horizon_list = [1, 6, 12, 24]

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse = calculate_rmse

def plot_prediction_series(true_values, predicted_values, plot_title, output_path):
    true_array = np.asarray(true_values)
    pred_array = np.asarray(predicted_values)
    n_samples = min(len(true_array), len(pred_array))
    
    plt.figure(figsize=(9, 3))
    plt.plot(true_array[:n_samples], label="True", linewidth=1.5)
    plt.plot(pred_array[:n_samples], label="Pred", linewidth=1.5)
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def prepare_tabular_data(train_data, test_data, target_column, forecast_horizon, feature_columns):
    X_train = train_data[feature_columns].copy()
    y_train = train_data[target_column].shift(-forecast_horizon).dropna()
    X_train = X_train.iloc[:len(y_train)]
    
    X_test = test_data[feature_columns].copy()
    y_test = test_data[target_column].shift(-forecast_horizon).dropna()
    X_test = X_test.iloc[:len(y_test)]
    
    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None, None
    
    scaler_obj = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler_obj.fit_transform(X_train), 
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler_obj.transform(X_test), 
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def prepare_sequence_data(train_data, test_data, target_column, forecast_horizon, feature_columns, lookback_window=48):
    max_horizon = forecast_horizon
    
    def build_sequences(dataframe):
        dataframe = dataframe.reset_index(drop=True)
        feature_matrix = dataframe[feature_columns].values
        target_values = dataframe[target_column].values
        n_rows = len(dataframe)
        
        X_sequences = []
        y_targets = []
        
        for i in range(lookback_window - 1, n_rows - max_horizon):
            seq_start = i - lookback_window + 1
            seq_end = i + 1
            X_sequences.append(feature_matrix[seq_start:seq_end])
            y_targets.append(target_values[i + forecast_horizon])
        
        if not X_sequences:
            return None, None
        
        X_array = np.stack(X_sequences, axis=0)
        y_array = np.array(y_targets)
        return X_array, y_array
    
    X_train_seq, y_train_seq = build_sequences(train_data)
    X_test_seq, y_test_seq = build_sequences(test_data)
    
    if X_train_seq is None or X_test_seq is None:
        return None, None, None, None
    
    batch_size, seq_length, num_features = X_train_seq.shape
    scaler_obj = StandardScaler()
    
    X_train_flat = scaler_obj.fit_transform(X_train_seq.reshape(batch_size, seq_length * num_features))
    X_test_flat = scaler_obj.transform(X_test_seq.reshape(X_test_seq.shape[0], seq_length * num_features))
    
    X_train_seq = X_train_flat.reshape(batch_size, seq_length, num_features)
    X_test_seq = X_test_flat.reshape(X_test_seq.shape[0], seq_length, num_features)
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq

if has_pytorch:
    
    class AttentionLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=1):
            super().__init__()
            self.lstm_layer = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.attention_layer = nn.Linear(hidden_dim, 1)
            self.output_layer = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm_layer(x)
            attention_weights = self.attention_layer(lstm_out)
            attention_scores = torch.softmax(attention_weights, dim=1)
            context_vector = (attention_scores * lstm_out).sum(dim=1)
            output = self.output_layer(context_vector).squeeze(-1)
            return output
    
    class TemporalConvNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.conv_layer_1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
            self.conv_layer_2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
            self.activation = nn.ReLU()
            self.output_layer = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.activation(self.conv_layer_1(x))
            x = self.activation(self.conv_layer_2(x))
            x = x.mean(dim=2)
            output = self.output_layer(x).squeeze(-1)
            return output
    
    class TransformerTimeSeries(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu",
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_layer = nn.Linear(d_model, 1)
        
        def forward(self, x):
            x = self.input_projection(x)
            encoded = self.transformer_encoder(x)
            context = encoded.mean(dim=1)
            output = self.output_layer(context).squeeze(-1)
            return output
    
    class TemporalFusionTransformer(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, hidden_dim=64):
            super().__init__()
            self.variable_projection = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                activation="relu",
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.gating_mechanism = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            self.output_layer = nn.Linear(d_model, 1)
        
        def forward(self, x):
            projected = self.variable_projection(x)
            encoded = self.transformer_encoder(projected)
            gate_values = self.gating_mechanism(encoded)
            gated_output = gate_values * encoded
            context = gated_output.mean(dim=1)
            output = self.output_layer(context).squeeze(-1)
            return output
    
    class InformerModel(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu",
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_layer = nn.Linear(d_model, 1)
        
        def forward(self, x):
            x = self.input_projection(x)
            seq_length = x.size(1)
            if seq_length > 16:
                random_indices = torch.randperm(seq_length)[:max(8, seq_length // 3)]
                random_indices, _ = torch.sort(random_indices)
                x = x[:, random_indices, :]
            encoded = self.transformer_encoder(x)
            context = encoded.mean(dim=1)
            output = self.output_layer(context).squeeze(-1)
            return output
    
    class AutoformerModel(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu",
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_layer = nn.Linear(d_model, 1)
        
        def forward(self, x):
            x = self.input_projection(x)
            encoded = self.transformer_encoder(x)
            attention_matrix = (encoded @ encoded.transpose(1, 2)).mean(dim=1)
            attention_weights = torch.softmax(attention_matrix, dim=-1)
            context = (attention_weights.unsqueeze(-1) * encoded).sum(dim=1)
            output = self.output_layer(context).squeeze(-1)
            return output
    
    def train_deep_model(model_instance, X_train_tensor, y_train_tensor, X_test_tensor, device_name, num_epochs=10, batch_size_value=128, learning_rate_value=1e-3):
        model_instance.to(device_name)
        
        X_train_t = torch.tensor(X_train_tensor, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_tensor, dtype=torch.float32)
        
        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=batch_size_value, shuffle=True)
        
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate_value)
        loss_function = nn.MSELoss()
        
        model_instance.train()
        for epoch_idx in range(num_epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device_name)
                batch_y = batch_y.to(device_name)
                optimizer.zero_grad()
                predictions = model_instance(batch_x)
                loss_value = loss_function(predictions, batch_y)
                loss_value.backward()
                optimizer.step()
        
        model_instance.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_tensor, dtype=torch.float32).to(device_name)
            test_predictions = model_instance(X_test_t).cpu().numpy()
        
        return test_predictions

def remove_outliers_iqr_method(dataframe, target_columns=target_list):
    cleaned_dataframe = dataframe.copy()
    
    for column_name in target_columns:
        if column_name not in cleaned_dataframe.columns:
            continue
        
        q1_value = cleaned_dataframe[column_name].quantile(0.25)
        q3_value = cleaned_dataframe[column_name].quantile(0.75)
        iqr_value = q3_value - q1_value
        lower_bound = q1_value - 1.5 * iqr_value
        upper_bound = q3_value + 1.5 * iqr_value
        
        rows_before = len(cleaned_dataframe)
        cleaned_dataframe = cleaned_dataframe[
            (cleaned_dataframe[column_name] >= lower_bound) & 
            (cleaned_dataframe[column_name] <= upper_bound)
        ]
        rows_after = len(cleaned_dataframe)
        print(f"[CLEAN] {column_name}: removed {rows_before - rows_after} rows")
    
    return cleaned_dataframe

clean_dataset_with_iqr = remove_outliers_iqr_method

def run_regression_models(train_dataframe, test_dataframe, feature_column_list, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    horizon_values = horizon_list
    target_values = target_list
    sequence_lookback = 48
    
    if has_pytorch and torch.cuda.is_available():
        compute_device = "cuda"
    else:
        compute_device = "cpu"
    print(f"[INFO] Using device = {compute_device}")
    
    all_results_list = []
    
    for target_variable in target_values:
        if target_variable not in train_dataframe.columns or target_variable not in test_dataframe.columns:
            print(f"[WARN] missing target {target_variable}, skip")
            continue
        
        for horizon_value in horizon_values:
            X_train_tab, y_train_tab, X_test_tab, y_test_tab = prepare_tabular_data(
                train_dataframe, test_dataframe, target_variable, horizon_value, feature_column_list
            )
            
            if X_train_tab is None:
                print(f"[WARN] empty supervised set for {target_variable}+{horizon_value}h, skip")
                continue
            
            prediction_dict = {}
            baseline_y_test = y_test_tab
            
            lgbm_model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            lgbm_model.fit(X_train_tab, y_train_tab)
            lgbm_preds = lgbm_model.predict(X_test_tab)
            prediction_dict["LightGBM"] = pd.Series(
                lgbm_preds, 
                index=baseline_y_test.index
            )
            
            rf_model = RandomForestRegressor(n_estimators=200, random_state=0)
            rf_model.fit(X_train_tab, y_train_tab)
            prediction_dict["RF"] = pd.Series(
                rf_model.predict(X_test_tab), 
                index=baseline_y_test.index
            )
            
            lr_model = LinearRegression()
            lr_model.fit(X_train_tab, y_train_tab)
            prediction_dict["Linear"] = pd.Series(
                lr_model.predict(X_test_tab), 
                index=baseline_y_test.index
            )
            
            if has_xgboost:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    tree_method="hist",
                )
                xgb_model.fit(X_train_tab, y_train_tab)
                prediction_dict["XGBoost"] = pd.Series(
                    xgb_model.predict(X_test_tab), 
                    index=baseline_y_test.index
                )
            
            if has_pytorch:
                X_train_seq, y_train_seq, X_test_seq, y_test_seq = prepare_sequence_data(
                    train_dataframe, test_dataframe, target_variable, horizon_value, 
                    feature_column_list, lookback_window=sequence_lookback
                )
            else:
                X_train_seq = y_train_seq = X_test_seq = y_test_seq = None
            
            sequence_data_valid = has_pytorch and X_train_seq is not None and X_test_seq is not None
            
            if sequence_data_valid:
                input_dimension = X_train_seq.shape[2]
                
                model_dict = {
                    "AttnLSTM": AttentionLSTM(input_dim=input_dimension),
                    "TCN": TemporalConvNet(input_dim=input_dimension),
                    "Transformer": TransformerTimeSeries(input_dim=input_dimension),
                    "TFT": TemporalFusionTransformer(input_dim=input_dimension),
                    "Informer": InformerModel(input_dim=input_dimension),
                    "Autoformer": AutoformerModel(input_dim=input_dimension),
                }
                
                for model_name, model_instance in model_dict.items():
                    try:
                        pred_array = train_deep_model(
                            model_instance, X_train_seq, y_train_seq, X_test_seq, compute_device
                        )
                        min_length = min(len(pred_array), len(baseline_y_test))
                        aligned_indices = baseline_y_test.index[-min_length:]
                        prediction_dict[model_name] = pd.Series(
                            pred_array[-min_length:], 
                            index=aligned_indices
                        )
                    except Exception as e:
                        print(f"[ERROR] Failed to train {model_name}: {e}")
                        continue
            
            score_dict = {}
            for model_name_key, pred_series in prediction_dict.items():
                common_indices = baseline_y_test.index.intersection(pred_series.index)
                if len(common_indices) == 0:
                    continue
                rmse_value = calculate_rmse(
                    baseline_y_test.loc[common_indices], 
                    pred_series.loc[common_indices]
                )
                score_dict[model_name_key] = rmse_value
                all_results_list.append({
                    "target": target_variable,
                    "horizon": horizon_value,
                    "model": model_name_key,
                    "rmse": rmse_value,
                })
            
            blend_model_list = [m for m in ["LightGBM", "RF", "Linear", "XGBoost"] if m in prediction_dict]
            if len(blend_model_list) >= 2:
                common_indices_for_blend = baseline_y_test.index
                for model_name in blend_model_list:
                    common_indices_for_blend = common_indices_for_blend.intersection(
                        prediction_dict[model_name].index
                    )
                if len(common_indices_for_blend) > 0:
                    stacked_predictions = np.stack([
                        prediction_dict[m].loc[common_indices_for_blend].values 
                        for m in blend_model_list
                    ], axis=0)
                    blended_predictions = stacked_predictions.mean(axis=0)
                    prediction_dict["BlendMean"] = pd.Series(
                        blended_predictions, 
                        index=common_indices_for_blend
                    )
                    blend_rmse = calculate_rmse(
                        baseline_y_test.loc[common_indices_for_blend], 
                        prediction_dict["BlendMean"].loc[common_indices_for_blend]
                    )
                    score_dict["BlendMean"] = blend_rmse
                    all_results_list.append({
                        "target": target_variable,
                        "horizon": horizon_value,
                        "model": "BlendMean",
                        "rmse": blend_rmse,
                    })
            
            if not score_dict:
                print(f"[WARN] no valid scores for {target_variable}+{horizon_value}h")
                continue
            
            best_model_name = min(score_dict, key=score_dict.get)
            print(f"[REG] {target_variable}+{horizon_value}h best = {best_model_name} RMSE={score_dict[best_model_name]:.4f}")
            
            best_prediction_series = prediction_dict[best_model_name]
            plot_indices = baseline_y_test.index.intersection(best_prediction_series.index)
            if len(plot_indices) > 0:
                plot_prediction_series(
                    baseline_y_test.loc[plot_indices],
                    best_prediction_series.loc[plot_indices],
                    f"{target_variable} +{horizon_value}h {best_model_name}",
                    os.path.join(output_directory, f"{target_variable}_{horizon_value}h_{best_model_name}.png"),
                )
            
            for model_name_key, pred_series in prediction_dict.items():
                save_indices = baseline_y_test.index.intersection(pred_series.index)
                if len(save_indices) == 0:
                    continue
                prediction_dataframe = pd.DataFrame({
                    "y_true": baseline_y_test.loc[save_indices].values,
                    "y_pred": pred_series.loc[save_indices].values,
                })
                prediction_dataframe.to_csv(
                    os.path.join(output_directory, f"{target_variable}_{horizon_value}h_{model_name_key}_pred.csv"),
                    index=False,
                )
    
    metrics_dataframe = pd.DataFrame(all_results_list)
    metrics_output_path = os.path.join(output_directory, "metrics.csv")
    metrics_dataframe.to_csv(metrics_output_path, index=False)
    print(f"[INFO] Saved metrics to {metrics_output_path}")

def run_regression_models_with_cleaning(train_dataframe, test_dataframe, feature_column_list, root_output_directory):
    base_output_dir = os.path.join(root_output_directory, "base")
    clean_output_dir = os.path.join(root_output_directory, "clean")
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(clean_output_dir, exist_ok=True)
    
    print("Running regression models on RAW data ...")
    run_regression_models(train_dataframe, test_dataframe, feature_column_list, base_output_dir)
    
    print("Cleaning train data with IQR for all targets ...")
    train_cleaned = remove_outliers_iqr_method(train_dataframe)
    
    print("Running regression models on CLEANED data ...")
    run_regression_models(train_cleaned, test_dataframe, feature_column_list, clean_output_dir)

def run_multihorizon_transformer(train_dataframe, test_dataframe, feature_column_list, output_directory):
    if not has_pytorch:
        print("[WARN] PyTorch not available, skipping multi-horizon Transformer.")
        return
    
    print("[WARN] run_multihorizon_transformer is not fully implemented")
    pass
