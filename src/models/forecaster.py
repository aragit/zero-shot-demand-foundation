import torch
import yaml
import numpy as np
from chronos import BaseChronosPipeline

class ZeroShotForecastingEngine:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.device = self.config.get("infrastructure", {}).get("device_map", "cpu")
        self._init_chronos()

    def _init_chronos(self):
        repo_id = self.config.get("models", {}).get("chronos", {}).get("repo_id", "amazon/chronos-2")
        self.chronos_pipeline = BaseChronosPipeline.from_pretrained(
            repo_id, device_map=self.device, dtype=torch.float32
        )

    def predict_chronos(self, input_payload) -> dict:
        # FIX: Chronos-2 strictly requires a 3D tensor: (n_series, n_variates, history_length)
        target_series = torch.tensor(input_payload.target_series, dtype=torch.float32).view(1, 1, -1)
        horizon = input_payload.horizon

        with torch.no_grad():
            output_tensor = self.chronos_pipeline.predict(inputs=target_series, prediction_length=horizon)
        
        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.tensor(np.array(output_tensor))
        
        # FIX: Robust, shape-agnostic parsing of Chronos-2 outputs
        if output_tensor.ndim == 3:
            # Shape: (n_series, n_variates, prediction_length) -> Point forecast directly
            median_forecast = output_tensor[0, 0, :].cpu().numpy()
        elif output_tensor.ndim == 4:
            # Shape: (n_series, num_samples/quantiles, n_variates, prediction_length)
            shape = output_tensor.shape
            if shape[1] != 1 and shape[2] == 1:
                # Compute median across the samples dimension (dim 1)
                median_tensor = torch.median(output_tensor, dim=1).values
                median_forecast = median_tensor[0, 0, :].cpu().numpy()
            elif shape[2] != 1 and shape[1] == 1:
                # Compute median across the quantiles dimension (dim 2)
                median_tensor = torch.median(output_tensor, dim=2).values
                median_forecast = median_tensor[0, 0, :].cpu().numpy()
            else:
                median_forecast = output_tensor[0, shape[1] // 2, 0, :].cpu().numpy()
        else:
            median_forecast = output_tensor.flatten().cpu().numpy()[:horizon]

        return {
            "median": median_forecast,
            "model_identifier": self.config.get("models", {}).get("chronos", {}).get("repo_id", "amazon/chronos-2")
        }
