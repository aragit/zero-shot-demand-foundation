from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

class TimeSeriesInputPayload(BaseModel):
    """
    Validates and enforces strict alignment for incoming time-series telemetry.
    Designed for In-Context Fine-Tuning (ICF) configurations.
    """
    target_series: List[float] = Field(
        ..., 
        description="Historical target demand observations vector."
    )
    price_index: Optional[List[float]] = Field(
        None, 
        description="Exogenous price elasticity mapping aligned chronologically with target + horizon."
    )
    promo_flag: Optional[List[int]] = Field(
        None, 
        description="Binary flags mapping active promotion/marketing events."
    )
    forecast_horizon: int = Field(
        default=24, 
        ge=1, 
        le=1024, 
        description="The number of future timesteps to project forward."
    )

    @field_validator("target_series")
    @classmethod
    def validate_context_bounds(cls, v: List[float]) -> List[float]:
        if len(v) < 16:
            raise ValueError("Context length too shallow. Minimum 16 timesteps required for zero-shot attention maps.")
        if len(v) > 16000:
            raise ValueError("Context bounds error. Exceeds max 16,000 timestep framework limit.")
        return v

    @model_validator(mode="after")
    def verify_exogenous_alignment(self) -> "TimeSeriesInputPayload":
        expected_exo_len = len(self.target_series) + self.forecast_horizon
        
        if self.price_index is not None:
            if len(self.price_index) != expected_exo_len:
                raise ValueError(
                    f"Price array misalignment. Expected {expected_exo_len} elements "
                    f"(Context: {len(self.target_series)} + Horizon: {self.forecast_horizon}), "
                    f"got {len(self.price_index)}."
                )
                
        if self.promo_flag is not None:
            if len(self.promo_flag) != expected_exo_len:
                raise ValueError(
                    f"Promo flag array misalignment. Expected {expected_exo_len} elements, got {len(self.promo_flag)}."
                )
            if not all(flag in (0, 1) for flag in self.promo_flag):
                raise ValueError("Promo flags must be strictly binary integers (0 or 1).")
                
        return self


class ForecastOutputPayload(BaseModel):
    """
    Enforces deterministic validation rules on predicted quantile values 
    emitted by zero-shot foundation heads.
    """
    model_identifier: str = Field(..., description="The explicit Hugging Face model key used for inference.")
    mean_prediction: List[float] = Field(..., description="Calculated target mean array over the forecasted horizon.")
    p10_quantile: Optional[List[float]] = Field(None, description="Lower bound variance forecast profile.")
    p90_quantile: Optional[List[float]] = Field(None, description="Upper bound variance forecast profile.")
    horizon_length: int = Field(..., description="Validated step length matching requested bounds.")

    @model_validator(mode="after")
    def verify_prediction_dimensions(self) -> "ForecastOutputPayload":
        actual_len = len(self.mean_prediction)
        if actual_len != self.horizon_length:
            raise ValueError(f"Output mismatch. Mean array length ({actual_len}) does not equal target horizon ({self.horizon_length}).")
        return self
