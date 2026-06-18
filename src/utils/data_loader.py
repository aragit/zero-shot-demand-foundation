import numpy as np
import pandas as pd
from typing import Dict, Any
from src.schemas.payloads import TimeSeriesInputPayload

class DemandDataEngine:
    """
    Handles streaming extraction and serialization of historical demand profiles
    into structured Pydantic input payloads.
    """
    def __init__(self, storage_root: str = "data/"):
        self.storage_root = storage_root

    def generate_synthetic_telemetry(self, context_length: int = 256, horizon: int = 24) -> TimeSeriesInputPayload:
        """
        Generates realistic volatile enterprise demand telemetry with aligned exogenous variables.
        """
        np.random.seed(42)
        total_steps = context_length + horizon
        
        # Build base timeline with seasonality and noise
        time_axis = np.linspace(0, 50, total_steps)
        base_demand = 150 + (30 * np.sin(time_axis)) + (10 * np.cos(time_axis / 2))
        noise = np.random.normal(0, 12, total_steps)
        target_series = (base_demand + noise).clip(min=10).tolist()
        
        # Build exogenous pricing index (simulating mid-timeline pricing hike)
        price_index = np.ones(total_steps) * 19.99
        price_index[context_length // 2:] = 24.99
        price_index = price_index.tolist()
        
        # Build binary promotional markers (sporadic marketing bursts)
        promo_flag = np.zeros(total_steps, dtype=int)
        promo_indices = np.random.choice(total_steps, size=total_steps // 10, replace=False)
        promo_flag[promo_indices] = 1
        promo_flag = promo_flag.tolist()

        # Slice target to match context length (the model shouldn't see future values)
        context_target = target_series[:context_length]

        return TimeSeriesInputPayload(
            target_series=context_target,
            price_index=price_index,
            promo_flag=promo_flag,
            forecast_horizon=horizon
        )
