# 🚗🚗 Automotive Data Science Project — Car Price Prediction Using Machine Learning

> Predicting automobile prices using regression models with an interactive Streamlit dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Project Structure](#project-structure)
5. [Models Used](#models-used)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [Dashboard Preview](#dashboard-preview)
9. [Author](#author)

---

## 🎯 Project Overview

This project builds an end-to-end machine learning pipeline to predict automobile prices based on technical and categorical features. It includes exploratory data analysis, feature engineering, model training, evaluation, and deployment as an interactive Streamlit dashboard.

---

## ❓ Problem Statement

A Chinese automobile company wants to enter the US market. They need to understand which variables significantly affect car pricing in the American market, and how well those variables describe the price of a car.

**Goal:** Build a regression model to predict car prices and identify the most influential features.

---

## 📊 Dataset Description

- **Source:** [Kaggle — Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
- **Records:** 205 cars
- **Features:** 26 columns (numerical + categorical)
- **Target:** `price` (USD)

### Key Features
| Feature | Description |
|---|---|
| `CarName` | Brand and model |
| `enginesize` | Engine displacement (cc) |
| `horsepower` | Engine power (hp) |
| `curbweight` | Vehicle weight (lbs) |
| `fueltype` | gas / diesel |
| `carbody` | sedan, hatchback, wagon, etc. |
| `drivewheel` | fwd / rwd / 4wd |
| `citympg` | City fuel efficiency |
| `price` | **Target variable** |

---

## 📁 Project Structure

```
car-price-prediction-ml/
│
├── app.py                              # Streamlit dashboard
├── car_prices.py                       # Core ML pipeline
├── Car_Price_Prediction_EDA_Modeling.ipynb  # Full EDA + modeling notebook
├── requirements.txt                    # Dependencies
├── data/
│   └── README.md                       # Dataset download instructions
└── README.md
```

---

## 🤖 Models Used

| Model | R² Score |
|---|---|
| Random Forest | ~0.96 |
| Gradient Boosting | ~0.95 |
| Ridge Regression | ~0.85 |
| Linear Regression | ~0.84 |

---

## 📈 Results

- **Best Model:** Random Forest Regressor
- **R² Score:** 0.96
- **MAE:** ~$1,100
- **Top Features:** engine size, horsepower, curb weight, car length

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/shaikshireen01-MS/car-price-prediction-ml.git
cd car-price-prediction-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
See `data/README.md` for instructions.

### 4. Run the Streamlit dashboard
```bash
streamlit run app.py
```

---

## 📊 Dashboard Preview

The interactive dashboard includes three tabs:
- **🔮 Predict Price** — Configure car features and get instant price prediction
- **📊 Data Explorer** — Visualize price distributions, brand comparisons, scatter plots
- **🏆 Model Comparison** — Compare all models, feature importances, actual vs predicted

---

## 👩‍💻 Author

**Shaik Shireen**
M.Sc. Data Science — University of Europe for Applied Sciences, Potsdam
[GitHub](https://github.com/shaikshireen01-MS)

---

## 📄 License

This project is licensed under the MIT License.
