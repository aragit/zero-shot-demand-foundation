
# Zero-Shot Demand Foundation Pipeline

A high-performance forecasting engine utilizing **Amazon Chronos-2** language modeling architectures for zero-shot time-series forecasting on high-volume retail demand datasets.

## 🧠 The Problem & The Solution
Historically, enterprise supply chain forecasting required massive, heavily engineered ML pipelines. Competitions were completely dominated by complex **LightGBM, XGBoost, and CatBoost** models packed with thousands of rolling lag variables, rolling means, and cross-sectional target encodings. These traditional models require constant retraining and struggle to adapt to sudden market shocks without explicit historical mapping.

**The Solution:** By building this specific foundation model repository, we are taking a fundamentally different approach. Instead of training discrete models to learn specific store dynamics, we are testing if pre-trained attention heads can read the statistical signature of a history array in-context and accurately extrapolate it *without any explicit backpropagation updates or local fine-tuning*.

---

## 🏆 The Core Benchmark: M5 Forecasting
The architecture we initialized maps directly to the most rigorous and foundational supply chain forecasting benchmark in data science history: The **M5 Forecasting Challenge** hosted by the Makridakis Open Forecasting Center (MOFC) alongside Walmart.

Because we engineered our repo using a dual-track output profile (mean prediction + quantile bands), our codebase spans across both companion tracks of this historical benchmark.

### 1. M5 Forecasting - Accuracy Track
* **The Challenge:** Predict the exact daily point forecasts of 3,049 retail products across 10 Walmart stores in 3 US states for a strict 28-day forward horizon.
* **Our Implementation:** Our pipeline relies on `target_series` context matrices and evaluates zero-shot point performance against this exact 28-day boundary profile using WAPE and RMSSE.

### 2. M5 Forecasting - Uncertainty Track
* **The Challenge:** Instead of a single number, guess the 9 distinct quantile boundaries (including the 10th and 90th percentiles) of the upcoming demand track.
* **Our Implementation:** Traditional boosting trees struggle with raw native distribution variance without massive custom loss engineering (like asymmetric Pinball Loss). Chronos-2 natively emits sample trajectories, fitting the uncertainty profile out-of-the-box. Our robust output parser dynamically maps these variable quantile outputs back to a normalized evaluation state.

---

## 🗺️ Data Architecture & Kaggle Mapping
The raw dataset layout from the M5 competition translates directly into our data structure and Pydantic validator payloads:

| Kaggle Competition File | Our Repository Target | Purpose |
| :--- | :--- | :--- |
| `sales_train_evaluation.csv` | `target_series` | **The Context:** Daily sales columns extending over 1,941 days. Safely windowed to respect 16k-timestep attention limits. |
| `sell_prices.csv` | `price_index` | **Dynamic Elasticity:** Walmart's weekly average prices per product/store, providing elasticity signals across promotional horizons. |
| `calendar.csv` | `promo_flag` | **Binary Event States:** Structural shocks like SNAP food stamps allowances, sporting events, and cultural holidays mapped to 0/1 matrices. |

---

## 🏪 Secondary Validation: Corporación Favorita
To validate the model on an open, rolling leaderboard without closed deadlines, this pipeline is also compatible with the **Store Sales - Time Series Forecasting** benchmark (Corporación Favorita Retail Data).
* **The Value:** It introduces hyper-volatile local inflation markers and complex regional Ecuadorian holidays. It is an excellent validation playground for verifying if foundation models can generalize zero-shot across distinct international locales without weight adjustments.

---


## 🚀 Key Features
* **Zero-Shot Evaluation:** Leverages pre-trained transformer weights from `amazon/chronos-2` without local parameter fine-tuning.
* **Chronos-2 Tensor Integration:** Natively implements the strict 3D tensor mapping `(n_series, n_variates, history_length)` mandated by multivariate-ready architectures.
* **Active High-Volume Filter:** Avoids historical stockout/discontinuation traps by isolating items based on recent sales velocity.
* **Robust Output Parser:** Dynamically maps point forecasts, raw trajectories, or variable quantile outputs back to a normalized evaluation state.

---


## ⚡ Quick Start

1. Prerequisites & Installation
Ensure you have PyTorch, Hugging Face Transformers, and the Amazon Chronos package installed:

```
pip install torch transformers pandas numpy pyyaml
pip install amazon-chronos
```

2. Dataset Setup
Download the M5 Forecasting competition data and place the evaluation file into the local structure:

```
mkdir -p data
```

Place sales_train_evaluation.csv inside the data/ folder

3. Execution
Run the evaluation pipeline to test zero-shot capabilities against an active, high-volume item:

```
python main.py
```

## 📊 Evaluation Framework
The system uses metrics aligned with the M5 Competition framework:

- WAPE (Weighted Absolute Percentage Error): Assesses general accuracy across the forecasting horizon.

- RMSSE (Root Mean Squared Scaled Error): Evaluates model variance relative to a historical naive baseline. An RMSSE < 1.0 indicates the model is successfully capturing structural seasonal variations.
