"""
Car Price Prediction — Interactive Streamlit Dashboard
Author: Shaik Shireen | M.Sc. Data Science, UE Potsdam
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

    .main { background-color: #0d0d0d; color: #f0f0f0; }
    .stApp { background-color: #0d0d0d; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #e94560; font-family: 'Syne', sans-serif; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }

    .prediction-box {
        background: linear-gradient(135deg, #e94560 0%, #c0392b 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-price { font-size: 3rem; font-weight: 800; color: white; font-family: 'Syne', sans-serif; }
    .prediction-label { font-size: 1rem; color: rgba(255,255,255,0.8); }

    .stSelectbox > div > div { background-color: #1a1a2e; border: 1px solid #333; }
    .stSlider > div > div { background-color: #e94560; }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white; border: none; border-radius: 8px;
        font-family: 'Syne', sans-serif; font-weight: 700;
        padding: 12px 32px; width: 100%; font-size: 1rem;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

    .sidebar .sidebar-content { background-color: #111; }
    div[data-testid="stSidebarContent"] { background-color: #111; }
</style>
""", unsafe_allow_html=True)

# ── Data generation (mirrors typical Kaggle CarPrice dataset structure) ───────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 205

    car_names = [
        "alfa-romero giulia", "alfa-romero stelvio", "audi 100", "audi 200",
        "bmw 320i", "bmw 530i", "chevrolet impala", "dodge colt", "honda accord",
        "honda civic", "isuzu MU-X", "jaguar xf", "mazda 626", "mercedes-benz 190",
        "mitsubishi galant", "nissan altima", "peugot 504", "plymouth acclaim",
        "porsche 911", "renault 12tl", "saab 900", "subaru legacy", "toyota camry",
        "toyota corolla", "volkswagen jetta", "volvo 245", "volvo 264",
    ]
    fuel_types = ["gas", "diesel"]
    aspiration_types = ["std", "turbo"]
    body_styles = ["convertible", "hardtop", "hatchback", "sedan", "wagon"]
    drive_wheels = ["4wd", "fwd", "rwd"]
    engine_locations = ["front", "rear"]
    engine_types = ["dohc", "ohc", "ohcv", "l", "rotor"]
    fuel_systems = ["mpfi", "2bbl", "4bbl", "idi", "spdi"]
    cylinder_numbers = ["two", "three", "four", "five", "six", "eight", "twelve"]

    data = {
        "CarName": np.random.choice(car_names, n),
        "fueltype": np.random.choice(fuel_types, n, p=[0.85, 0.15]),
        "aspiration": np.random.choice(aspiration_types, n, p=[0.80, 0.20]),
        "carbody": np.random.choice(body_styles, n, p=[0.06, 0.08, 0.34, 0.46, 0.06]),
        "drivewheel": np.random.choice(drive_wheels, n, p=[0.09, 0.60, 0.31]),
        "enginelocation": np.random.choice(engine_locations, n, p=[0.97, 0.03]),
        "enginetype": np.random.choice(engine_types, n),
        "fuelsystem": np.random.choice(fuel_systems, n),
        "cylindernumber": np.random.choice(cylinder_numbers, n, p=[0.02, 0.02, 0.55, 0.05, 0.24, 0.10, 0.02]),
        "wheelbase": np.random.uniform(86.6, 120.9, n).round(1),
        "carlength": np.random.uniform(141.1, 208.1, n).round(1),
        "carwidth": np.random.uniform(60.3, 72.3, n).round(1),
        "carheight": np.random.uniform(47.8, 59.8, n).round(1),
        "curbweight": np.random.randint(1488, 4066, n),
        "enginesize": np.random.randint(61, 326, n),
        "boreratio": np.random.uniform(2.54, 3.94, n).round(2),
        "stroke": np.random.uniform(2.07, 4.17, n).round(2),
        "compressionratio": np.random.uniform(7.0, 23.0, n).round(1),
        "horsepower": np.random.randint(48, 288, n),
        "peakrpm": np.random.randint(4150, 6600, n),
        "citympg": np.random.randint(13, 49, n),
        "highwaympg": np.random.randint(16, 54, n),
        "symboling": np.random.choice([-2, -1, 0, 1, 2, 3], n, p=[0.04, 0.10, 0.29, 0.26, 0.23, 0.08]),
        "doornumber": np.random.choice(["two", "four"], n, p=[0.35, 0.65]),
    }

    df = pd.DataFrame(data)
    # Derive price from features (realistic relationship)
    price = (
        df["enginesize"] * 85
        + df["horsepower"] * 120
        + df["curbweight"] * 4.5
        + df["wheelbase"] * 200
        - df["citympg"] * 150
        + (df["fueltype"] == "diesel").astype(int) * 2000
        + (df["aspiration"] == "turbo").astype(int) * 3500
        + (df["drivewheel"] == "rwd").astype(int) * 1800
        + (df["carbody"] == "convertible").astype(int) * 5000
        + np.random.normal(0, 2500, n)
    )
    df["price"] = np.clip(price, 5118, 45400).round(-2)
    return df

# ── Feature engineering & model training ─────────────────────────────────────
@st.cache_resource
def train_models(df):
    df = df.copy()

    # Extract brand from CarName
    df["brand"] = df["CarName"].apply(lambda x: x.split()[0])

    cat_cols = ["fueltype", "aspiration", "carbody", "drivewheel",
                "enginelocation", "enginetype", "fuelsystem",
                "cylindernumber", "doornumber", "brand"]

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        le_dict[col] = le

    feature_cols = [
        "fueltype_enc", "aspiration_enc", "carbody_enc", "drivewheel_enc",
        "enginelocation_enc", "enginetype_enc", "fuelsystem_enc",
        "cylindernumber_enc", "doornumber_enc", "brand_enc",
        "wheelbase", "carlength", "carwidth", "carheight",
        "curbweight", "enginesize", "boreratio", "stroke",
        "compressionratio", "horsepower", "peakrpm",
        "citympg", "highwaympg", "symboling",
    ]

    X = df[feature_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "y_test": y_test,
            "preds": preds,
        }

    return results, le_dict, feature_cols, X_test, y_test

# ── Load everything ───────────────────────────────────────────────────────────
df = load_data()
model_results, le_dict, feature_cols, X_test, y_test = train_models(df)
best_model_name = max(model_results, key=lambda k: model_results[k]["r2"])
best_model = model_results[best_model_name]["model"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#e94560; margin-bottom:0;'>🚗 Car Price Predictor</h1>
<p style='color:#888; font-size:1.1rem; margin-top:4px;'>
End-to-End ML Regression Pipeline with Interactive Dashboard
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Data Explorer", "🏆 Model Comparison"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Configure Your Car")
    col1, col2, col3 = st.columns(3)

    with col1:
        fuel_type = st.selectbox("Fuel Type", ["gas", "diesel"])
        aspiration = st.selectbox("Aspiration", ["std", "turbo"])
        carbody = st.selectbox("Body Style", ["sedan", "hatchback", "wagon", "convertible", "hardtop"])
        drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
        doornumber = st.selectbox("Doors", ["four", "two"])
        brand = st.selectbox("Brand", sorted(df["CarName"].apply(lambda x: x.split()[0]).unique()))

    with col2:
        horsepower = st.slider("Horsepower", 48, 290, 120)
        enginesize = st.slider("Engine Size (cc)", 61, 330, 130)
        curbweight = st.slider("Curb Weight (lbs)", 1488, 4070, 2500)
        wheelbase = st.slider("Wheelbase (in)", 86.6, 121.0, 98.4)
        cylindernumber = st.selectbox("Cylinders", ["four", "six", "eight", "two", "three", "five", "twelve"])

    with col3:
        citympg = st.slider("City MPG", 13, 50, 25)
        highwaympg = st.slider("Highway MPG", 16, 55, 30)
        compressionratio = st.slider("Compression Ratio", 7.0, 23.0, 9.0)
        boreratio = st.slider("Bore Ratio", 2.54, 3.94, 3.19)
        enginelocation = st.selectbox("Engine Location", ["front", "rear"])

    predict_btn = st.button("🔮 Predict Price")

    if predict_btn:
        # Build input row
        input_dict = {
            "fueltype": fuel_type, "aspiration": aspiration, "carbody": carbody,
            "drivewheel": drivewheel, "enginelocation": enginelocation,
            "enginetype": "ohc", "fuelsystem": "mpfi",
            "cylindernumber": cylindernumber, "doornumber": doornumber, "brand": brand,
            "wheelbase": wheelbase, "carlength": 174.0, "carwidth": 65.0,
            "carheight": 54.0, "curbweight": curbweight, "enginesize": enginesize,
            "boreratio": boreratio, "stroke": 3.4, "compressionratio": compressionratio,
            "horsepower": horsepower, "peakrpm": 5500, "citympg": citympg,
            "highwaympg": highwaympg, "symboling": 0,
        }

        row = {}
        cat_cols_map = ["fueltype", "aspiration", "carbody", "drivewheel", "enginelocation",
                        "enginetype", "fuelsystem", "cylindernumber", "doornumber", "brand"]
        for col in cat_cols_map:
            val = input_dict[col]
            le = le_dict[col]
            if val in le.classes_:
                row[col + "_enc"] = le.transform([val])[0]
            else:
                row[col + "_enc"] = 0

        num_cols = ["wheelbase", "carlength", "carwidth", "carheight", "curbweight",
                    "enginesize", "boreratio", "stroke", "compressionratio",
                    "horsepower", "peakrpm", "citympg", "highwaympg", "symboling"]
        for col in num_cols:
            row[col] = input_dict[col]

        input_df = pd.DataFrame([row])[feature_cols]
        predicted_price = best_model.predict(input_df)[0]

        st.markdown(f"""
        <div class='prediction-box'>
            <div class='prediction-label'>Estimated Market Price</div>
            <div class='prediction-price'>${predicted_price:,.0f}</div>
            <div class='prediction-label' style='margin-top:8px;'>
                Predicted using {best_model_name} (Best Model · R² = {model_results[best_model_name]['r2']:.3f})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Price range
        margin = model_results[best_model_name]["mae"]
        c1, c2 = st.columns(2)
        c1.metric("Lower Estimate", f"${max(0, predicted_price - margin):,.0f}")
        c2.metric("Upper Estimate", f"${predicted_price + margin:,.0f}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Dataset Overview")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Avg Price", f"${df['price'].mean():,.0f}")
    m3.metric("Min Price", f"${df['price'].min():,.0f}")
    m4.metric("Max Price", f"${df['price'].max():,.0f}")

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = px.histogram(df, x="price", nbins=30,
                            title="Price Distribution",
                            color_discrete_sequence=["#e94560"])
        fig1.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                           font_color="#ccc", title_font_color="#e94560")
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        df["brand"] = df["CarName"].apply(lambda x: x.split()[0])
        brand_avg = df.groupby("brand")["price"].mean().sort_values(ascending=False).head(12)
        fig2 = px.bar(brand_avg, title="Avg Price by Brand (Top 12)",
                      color=brand_avg.values,
                      color_continuous_scale=["#333", "#e94560"])
        fig2.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                           font_color="#ccc", title_font_color="#e94560",
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig3 = px.scatter(df, x="enginesize", y="price", color="fueltype",
                          title="Engine Size vs Price",
                          color_discrete_map={"gas": "#e94560", "diesel": "#0f3460"})
        fig3.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111", font_color="#ccc",
                           title_font_color="#e94560")
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.box(df, x="carbody", y="price", title="Price by Body Style",
                      color="carbody",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig4.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111", font_color="#ccc",
                           title_font_color="#e94560", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Raw Data")
    st.dataframe(df.head(50), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Performance Comparison")

    metrics_data = {
        "Model": list(model_results.keys()),
        "R² Score": [round(v["r2"], 4) for v in model_results.values()],
        "MAE ($)": [round(v["mae"], 0) for v in model_results.values()],
        "RMSE ($)": [round(v["rmse"], 0) for v in model_results.values()],
    }
    metrics_df = pd.DataFrame(metrics_data).sort_values("R² Score", ascending=False)
    metrics_df["Best"] = metrics_df["Model"] == best_model_name

    st.dataframe(metrics_df[["Model", "R² Score", "MAE ($)", "RMSE ($)"]],
                 use_container_width=True)

    fig5 = px.bar(metrics_df, x="Model", y="R² Score",
                  title="R² Score by Model",
                  color="R² Score", color_continuous_scale=["#333", "#e94560"],
                  text="R² Score")
    fig5.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig5.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                       font_color="#ccc", title_font_color="#e94560")
    st.plotly_chart(fig5, use_container_width=True)

    # Actual vs Predicted for best model
    best = model_results[best_model_name]
    fig6 = px.scatter(x=best["y_test"], y=best["preds"],
                      labels={"x": "Actual Price ($)", "y": "Predicted Price ($)"},
                      title=f"Actual vs Predicted — {best_model_name}",
                      color_discrete_sequence=["#e94560"])
    max_val = max(best["y_test"].max(), best["preds"].max())
    fig6.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                   line=dict(color="#555", dash="dash"))
    fig6.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                       font_color="#ccc", title_font_color="#e94560")
    st.plotly_chart(fig6, use_container_width=True)

    # Feature importance for RF
    rf = model_results["Random Forest"]["model"]
    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
    fig7 = px.bar(feat_imp, title="Top 12 Feature Importances (Random Forest)",
                  color=feat_imp.values, color_continuous_scale=["#333", "#e94560"])
    fig7.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                       font_color="#ccc", title_font_color="#e94560", showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)
