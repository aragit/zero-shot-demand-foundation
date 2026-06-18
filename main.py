import os
import pandas as pd
import numpy as np
from src.models.forecaster import ZeroShotForecastingEngine
from src.utils.metrics import MetricEvaluator

class TimeSeriesInputPayload:
    def __init__(self, target_series, horizon=24):
        self.target_series = target_series
        self.horizon = horizon

def main():
    print("======================================================================")
    print("INITIALIZING FIXED ZERO-SHOT EVALUATION PIPELINE")
    print("======================================================================\n")
    
    data_path = "data/sales_train_evaluation.csv"
    if not os.path.exists(data_path):
        print(f"[Error] Data missing at {data_path}.")
        return

    df = pd.read_csv(data_path)
    horizon = 24
    window_size = 128
    
    # Filter for active high-volume items over the current window
    active_window_sales = df.iloc[:, -window_size:]
    top_active_indices = active_window_sales.sum(axis=1).nlargest(100).index
    selected_idx = np.random.choice(top_active_indices)
    
    item_id = df.loc[selected_idx, 'item_id']
    store_id = df.loc[selected_idx, 'store_id']
    
    # Extract raw data points
    series = df.loc[selected_idx].iloc[6:].values.flatten().astype(float)[-window_size:]
    context, ground_truth = series[:-horizon], series[-horizon:]

    # Pass the raw context to our updated engine
    input_payload = TimeSeriesInputPayload(target_series=context.tolist(), horizon=horizon)
    forecaster = ZeroShotForecastingEngine(config_path="configs/model_config.yaml")
    results = forecaster.predict_chronos(input_payload)
    
    # Process outputs securely
    raw_forecast = np.array(results['median']).flatten()
    final_forecast = np.maximum(0, raw_forecast[:horizon])
    ground_truth_arr = np.array(ground_truth)
    
    # Calculate evaluation metrics
    evaluator = MetricEvaluator()
    wape = evaluator.calculate_wape(ground_truth_arr, final_forecast)
    rmsse = evaluator.calculate_rmsse(ground_truth_arr, final_forecast, context)
    
    print(f"Product Selected : {item_id} at Store {store_id}")
    print(f"Model ID         : {results.get('model_identifier', 'N/A')}")
    print("-" * 50)
    print(f"WAPE (Accuracy)  : {wape:.4f}%")
    print(f"RMSSE (M5 Std)   : {rmsse:.4f}")
    print("-" * 50)
    
    print(f"Context Mean     : {np.mean(context):.2f} units/day")
    print(f"Median Forecast  (First 5): {[round(float(x), 3) for x in final_forecast[:5]]}")
    print(f"Ground Truth     (First 5): {[round(float(x), 3) for x in ground_truth_arr[:5]]}")
    print("=" * 60)

if __name__ == "__main__":
    main()
