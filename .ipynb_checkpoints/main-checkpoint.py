import sys, os

# 自动把项目根目录加入 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 现在 from src.xxx 就可以被识别
from src.preprocess import (
    load_and_clean, impute_base, add_time_features,
    add_lag_ma_features, temporal_split, build_feature_list
)
from src.regression import run_regression_models
from src.classification import run_classification_models
from src.anomaly_detection import run_anomaly_detection
def debug_shape(name, df):
    print(f"[DEBUG] {name}: rows={len(df)}, cols={df.shape[1]}")
    if len(df) > 0:
        print(df.head(2))
    print("-" * 50)


DATA_PATH = "data/AirQualityUCI.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_END = "2004-12-31 23:00"
TEST_END = "2005-02-28 23:00"
TARGETS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
METEO = ["T", "RH", "AH"]
ALL_FEATURES_RAW = TARGETS + METEO

print("Loading and preprocessing data...")
df = load_and_clean(DATA_PATH)
debug_shape("After load_and_clean", df)

df = impute_base(df, ALL_FEATURES_RAW)
debug_shape("After impute_base", df)

df = add_time_features(df)
debug_shape("After add_time_features", df)

df = add_lag_ma_features(df, ALL_FEATURES_RAW, [1,6,12,24], [3,6,12,24])
debug_shape("After add_lag_ma_features", df)

# Build features BEFORE dropping NaN
feature_cols = build_feature_list(df)
print("[DEBUG] number of feature_cols =", len(feature_cols))

# Only drop rows that have NaN in important feature columns
df = df.dropna(subset=feature_cols).reset_index(drop=True)
debug_shape("After dropna(feature_cols)", df)

train, test = temporal_split(df, TRAIN_END, TEST_END)
debug_shape("Train", train)
debug_shape("Test", test)


# print("Running regression models...")
# run_regression_models(train, test, feature_cols, OUTPUT_DIR)
# from src.regression import run_multihorizon_transformer


# print("Running multi-horizon Transformer...")
# run_multihorizon_transformer(train, test, feature_cols, OUTPUT_DIR)

# from src.regression import run_regression_models_with_cleaning
# print("Running full regression pipeline with cleaning ...")
# run_regression_models_with_cleaning(train, test, feature_cols, OUTPUT_DIR)


# print("Running anomaly detection...")
# run_anomaly_detection(train, test, feature_cols, OUTPUT_DIR)


print("Running classification models...")
run_classification_models(train, test, feature_cols, OUTPUT_DIR)

print("All tasks completed.")
