"""
DISCO - Dynamic Instruction Simulation for Computer Organization
ai/predictor.py

The AI/ML component of DISCO. It is deliberately simple and explainable:
a Linear Regression model that learns the relationship between COA
performance parameters and execution time.

Central AI-based COA idea:

    CPU simulation  ->  measurable COA metrics  ->  ML training  ->  prediction

Features (inputs):
    - instruction_count
    - cpi
    - cache_hit_rate
    - clock_frequency_ghz

Target (output):
    - execution_time_us   (execution time in microseconds)

We use a standard train/test split and report Mean Absolute Error (MAE)
and R^2 so the model quality can be explained in a viva.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from analysis.dataset import load_dataset

# Feature columns fed into the model, and the target column it predicts.
FEATURE_COLUMNS: List[str] = [
    "instruction_count",
    "cpi",
    "cache_hit_rate",
    "clock_frequency_ghz",
]
TARGET_COLUMN: str = "execution_time_us"

# Minimum rows required to do a meaningful train/test split.
MIN_ROWS_FOR_TRAINING = 8


class PredictorError(Exception):
    """Raised when the model cannot be trained or used (e.g. too little data)."""


@dataclass
class TrainingResult:
    """Outcome of training the model, used for display in the UI."""

    mae: float                       # mean absolute error on the test set (us)
    r2: float                        # R^2 score on the test set
    n_train: int                     # number of training rows
    n_test: int                      # number of test rows
    coefficients: Dict[str, float]   # learned coefficient per feature
    intercept: float                 # learned intercept


class ExecutionTimePredictor:
    """A simple Linear Regression predictor for CPU execution time."""

    def __init__(self) -> None:
        self.model = LinearRegression()
        self.trained: bool = False
        self.result: TrainingResult | None = None

    def train(self, df: pd.DataFrame | None = None,
              test_size: float = 0.25, random_state: int = 42) -> TrainingResult:
        """Train the model on the dataset.

        Args:
            df: Dataset to train on. If None, loads the CSV dataset.
            test_size: Fraction reserved for testing.
            random_state: Seed for a reproducible split.

        Returns:
            A TrainingResult with error metrics and learned parameters.

        Raises:
            PredictorError: if there is not enough data to train.
        """
        if df is None:
            df = load_dataset()

        missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
        if missing:
            raise PredictorError(f"Dataset is missing required columns: {missing}")

        if len(df) < MIN_ROWS_FOR_TRAINING:
            raise PredictorError(
                f"Not enough data to train (have {len(df)} rows, "
                f"need at least {MIN_ROWS_FOR_TRAINING}). Generate more samples."
            )

        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = df[TARGET_COLUMN].to_numpy(dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred = self.model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        coeffs = {
            name: float(coef)
            for name, coef in zip(FEATURE_COLUMNS, self.model.coef_)
        }

        self.result = TrainingResult(
            mae=round(mae, 6),
            r2=round(r2, 4),
            n_train=len(X_train),
            n_test=len(X_test),
            coefficients={k: round(v, 6) for k, v in coeffs.items()},
            intercept=round(float(self.model.intercept_), 6),
        )
        return self.result

    def predict_one(self, metrics: Dict[str, float]) -> float:
        """Predict execution time (microseconds) for a single simulation.

        Args:
            metrics: A metrics dict containing the FEATURE_COLUMNS keys
                     (as returned by analysis.metrics.compute_metrics).

        Returns:
            Predicted execution time in microseconds.

        Raises:
            PredictorError: if the model has not been trained.
        """
        if not self.trained:
            raise PredictorError("Model must be trained before predicting.")
        features = np.array([[float(metrics[c]) for c in FEATURE_COLUMNS]])
        prediction = float(self.model.predict(features)[0])
        # Execution time cannot be negative; clamp for a sensible display value.
        return max(prediction, 0.0)


def prediction_error(actual_us: float, predicted_us: float) -> Dict[str, float]:
    """Compute absolute and percentage error between actual and predicted time."""
    abs_error = abs(actual_us - predicted_us)
    pct_error = (abs_error / actual_us * 100.0) if actual_us else 0.0
    return {
        "absolute_error_us": round(abs_error, 6),
        "percentage_error": round(pct_error, 2),
    }
