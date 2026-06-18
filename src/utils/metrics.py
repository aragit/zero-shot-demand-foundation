import numpy as np

class MetricEvaluator:
    def calculate_wape(self, actual, forecast):
        actual = np.array(actual)
        forecast = np.array(forecast)
        # Avoid division by zero
        denom = np.sum(np.abs(actual))
        if denom == 0: return 0.0
        return np.sum(np.abs(actual - forecast)) / denom * 100

    def calculate_rmsse(self, actual, forecast, history):
        """Root Mean Squared Scaled Error."""
        actual = np.array(actual)
        forecast = np.array(forecast)
        history = np.array(history)
        
        # Mean Squared Error of the forecast
        mse = np.mean((actual - forecast) ** 2)
        
        # Scaling factor: Mean Squared Error of a naive (lag-1) in-sample forecast
        scale_sq = np.mean(np.diff(history) ** 2)
        
        if scale_sq == 0: return 0.0
        
        return np.sqrt(mse / scale_sq)
