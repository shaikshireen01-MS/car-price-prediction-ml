"""
Car Price Prediction — Core ML Pipeline
Author: Shaik Shireen | M.Sc. Data Science, University of Europe for Applied Sciences, Potsdam
Description: End-to-end regression pipeline for automobile price prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH    = "CarPrice_Assignment.csv"
MODEL_DIR    = "model"
FIGURES_DIR  = "figures"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

CATEGORICAL_COLS = [
    "fueltype", "aspiration", "carbody", "drivewheel",
    "enginelocation", "enginetype", "fuelsystem",
    "cylindernumber", "doornumber", "brand"
]

FEATURE_COLS = [
    "fueltype_enc", "aspiration_enc", "carbody_enc", "drivewheel_enc",
    "enginelocation_enc", "enginetype_enc", "fuelsystem_enc",
    "cylindernumber_enc", "doornumber_enc", "brand_enc",
    "wheelbase", "carlength", "carwidth", "carheight",
    "curbweight", "enginesize", "boreratio", "stroke",
    "compressionratio", "horsepower", "peakrpm",
    "citympg", "highwaympg", "symboling",
]

TARGET_COL = "price"


# ── Utility ───────────────────────────────────────────────────────────────────
def make_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ── 1. Data Loading ───────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    """Load and return the raw dataset."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df):
    """Extract and clean features from raw dataframe."""
    df = df.copy()

    # Extract brand from CarName
    df["brand"] = df["CarName"].apply(lambda x: x.split()[0].lower())

    # Fix common brand name typos in this dataset
    brand_corrections = {
        "maxda"      : "mazda",
        "toyouta"    : "toyota",
        "vokswagen"  : "volkswagen",
        "vw"         : "volkswagen",
        "porcshce"   : "porsche",
        "alfa-romero": "alfa-romeo",
    }
    df["brand"] = df["brand"].replace(brand_corrections)

    print(f"Unique brands after cleaning: {sorted(df['brand'].unique())}")
    return df


# ── 3. Encoding ───────────────────────────────────────────────────────────────
def encode_features(df):
    """Label-encode categorical columns. Returns encoded df and encoder dict."""
    df = df.copy()
    le_dict = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict


# ── 4. Train / Test Split ─────────────────────────────────────────────────────
def split_data(df):
    """Split into train and test sets."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ── 5. Model Training ─────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    """Train multiple regression models and return fitted model dict."""
    models = {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression"  : Ridge(alpha=1.0),
        "Lasso Regression"  : Lasso(alpha=10.0),
        "Random Forest"     : RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }

    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name}")

    return models


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate all models and return results dataframe."""
    records = []

    for name, model in models.items():
        train_preds = model.predict(X_train)
        test_preds  = model.predict(X_test)
        cv_scores   = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        records.append({
            "Model"       : name,
            "Train R²"    : round(r2_score(y_train, train_preds), 4),
            "Test R²"     : round(r2_score(y_test,  test_preds),  4),
            "CV R² (mean)": round(cv_scores.mean(), 4),
            "MAE ($)"     : round(mean_absolute_error(y_test, test_preds), 0),
            "RMSE ($)"    : round(np.sqrt(mean_squared_error(y_test, test_preds)), 0),
        })

    results_df = pd.DataFrame(records).sort_values("Test R²", ascending=False)
    print("\n── Model Performance ──────────────────────────────────")
    print(results_df.to_string(index=False))
    return results_df


# ── 7. Visualizations ─────────────────────────────────────────────────────────
def plot_results(models, X_test, y_test, results_df):
    """Generate and save evaluation plots."""
    make_dirs()
    plt.style.use("dark_background")
    ACCENT = "#e94560"

    # 7a. R² comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(results_df["Model"], results_df["Test R²"],
                   color=ACCENT, alpha=0.85)
    ax.set_xlabel("Test R² Score", fontsize=12)
    ax.set_title("Model Comparison — Test R²", fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=4)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURES_DIR}/model_comparison.png")

    # 7b. Actual vs Predicted — best model
    best_name  = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    preds      = best_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, preds, alpha=0.7, color=ACCENT, edgecolors="none", s=50)
    lim = max(y_test.max(), preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "w--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Price ($)", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {best_name}\nR² = {results_df.iloc[0]['Test R²']:.4f}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURES_DIR}/actual_vs_predicted.png")

    # 7c. Feature importance — Random Forest
    if "Random Forest" in models:
        rf = models["Random Forest"]
        feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
        feat_imp = feat_imp.sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(feat_imp.index, feat_imp.values, color=ACCENT, alpha=0.85)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title("Top 15 Feature Importances (Random Forest)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIGURES_DIR}/feature_importance.png")


# ── 8. Save Best Model ────────────────────────────────────────────────────────
def save_best_model(models, results_df, le_dict):
    """Save the best performing model and encoders to disk."""
    make_dirs()
    best_name  = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    model_file = f"{MODEL_DIR}/best_model.pkl"
    enc_file   = f"{MODEL_DIR}/label_encoders.pkl"

    joblib.dump(best_model, model_file)
    joblib.dump(le_dict,    enc_file)
    print(f"\nBest model saved  → {model_file}")
    print(f"Encoders saved    → {enc_file}")
    return model_file


# ── 9. Predict Single Car ─────────────────────────────────────────────────────
def predict_price(input_dict, model_path=f"{MODEL_DIR}/best_model.pkl",
                  enc_path=f"{MODEL_DIR}/label_encoders.pkl"):
    """
    Predict price for a single car given a dict of raw feature values.

    Example input_dict:
    {
        'fueltype': 'gas', 'aspiration': 'std', 'carbody': 'sedan',
        'drivewheel': 'fwd', 'enginelocation': 'front', 'enginetype': 'ohc',
        'fuelsystem': 'mpfi', 'cylindernumber': 'four', 'doornumber': 'four',
        'brand': 'toyota', 'wheelbase': 98.4, 'carlength': 174.0,
        'carwidth': 65.0, 'carheight': 54.0, 'curbweight': 2500,
        'enginesize': 130, 'boreratio': 3.19, 'stroke': 3.4,
        'compressionratio': 9.0, 'horsepower': 120, 'peakrpm': 5500,
        'citympg': 25, 'highwaympg': 30, 'symboling': 0
    }
    """
    model   = joblib.load(model_path)
    le_dict = joblib.load(enc_path)

    row = {}
    for col in CATEGORICAL_COLS:
        val = input_dict[col]
        le  = le_dict[col]
        row[col + "_enc"] = le.transform([val])[0] if val in le.classes_ else 0

    num_cols = [c for c in FEATURE_COLS if not c.endswith("_enc")]
    for col in num_cols:
        row[col] = input_dict[col]

    X = pd.DataFrame([row])[FEATURE_COLS]
    predicted_price = model.predict(X)[0]
    return round(predicted_price, 2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Car Price Prediction — ML Pipeline")
    print("  Author: Shaik Shireen")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Encode
    df_enc, le_dict = encode_features(df)

    # 4. Split
    X_train, X_test, y_train, y_test = split_data(df_enc)

    # 5. Train
    models = train_models(X_train, y_train)

    # 6. Evaluate
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test)

    # 7. Visualize
    print("\nGenerating plots...")
    plot_results(models, X_test, y_test, results_df)

    # 8. Save best model
    save_best_model(models, results_df, le_dict)

    print("\n✓ Pipeline complete!")
    print(f"  Best Model : {results_df.iloc[0]['Model']}")
    print(f"  Test R²    : {results_df.iloc[0]['Test R²']}")
    print(f"  MAE        : ${results_df.iloc[0]['MAE ($)']:,.0f}")


if __name__ == "__main__":
    main()
