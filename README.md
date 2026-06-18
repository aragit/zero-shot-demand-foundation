# Zero-Shot Demand Foundation Pipeline

A high-performance forecasting engine utilizing **Amazon Chronos-2** language modeling architectures for zero-shot time-series forecasting on high-volume retail demand datasets (M5 Walmart Data).

## 🚀 Key Features
* **Zero-Shot Evaluation:** Leverages pre-trained transformer weights from `amazon/chronos-2` without local parameter fine-tuning.
* **Chronos-2 Tensor Integration:** Natively implements the strict 3D tensor mapping `(n_series, n_variates, history_length)` mandated by multivariate-ready architectures.
* **Active High-Volume Filter:** Avoids historical stockout/discontinuation traps by isolating items based on recent sales velocity.
* **Robust Output Parser:** Dynamically maps point forecasts, raw trajectories, or variable quantile outputs back to a normalized evaluation state.

---


## ⚡ Quick Start

1. Prerequisites & Installation
Ensure you have PyTorch, Hugging Face Transformers, and the Amazon Chronos package installed:


pip install torch transformers pandas numpy pyyaml
pip install amazon-chronos

2. Dataset Setup
Download the M5 Forecasting competition data and place the evaluation file into the local structure:


mkdir -p data
Place sales_train_evaluation.csv inside the data/ folder

3. Execution
Run the evaluation pipeline to test zero-shot capabilities against an active, high-volume item:


python main.py


## 📊 Evaluation Framework
The system uses metrics aligned with the M5 Competition framework:

- WAPE (Weighted Absolute Percentage Error): Assesses general accuracy across the forecasting horizon.

- RMSSE (Root Mean Squared Scaled Error): Evaluates model variance relative to a historical naive baseline. An RMSSE < 1.0 indicates the model is successfully capturing structural seasonal variations.


3. Save and exit nano:
   * Press **`Ctrl + O`** then press **`Enter`** (to write the file).
   * Press **`Ctrl + X`** (to close the editor).
