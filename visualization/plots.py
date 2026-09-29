"""
DISCO - Dynamic Instruction Simulation for Computer Organization
visualization/plots.py

Builds the graphs that communicate the COA results. Every plot has a clear
title, axis labels and (where useful) a legend. The functions return a
matplotlib Figure so the Streamlit UI can render them directly.

Provided plots:
    1. Configuration/workload vs Execution Time (across dataset samples)
    2. Actual Execution Time vs AI Predicted Execution Time
    3. (supporting) Cache Hit Rate vs Execution Time
    4. (supporting) Clock Frequency vs Execution Time

These plots directly illustrate core COA relationships: how frequency and
cache behaviour affect execution time, and how well the ML model predicts it.
"""

from __future__ import annotations

from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_workload_vs_execution_time(df: pd.DataFrame):
    """Bar/line chart of execution time across dataset simulation samples.

    X axis is the simulation sample index (each row = one workload/config),
    Y axis is execution time in microseconds.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    y = df["execution_time_us"].to_numpy()
    ax.plot(x, y, marker="o", linestyle="-", color="#2b6cb0", label="Execution Time")
    ax.set_title("Simulation Sample vs Execution Time")
    ax.set_xlabel("Simulation Sample #")
    ax.set_ylabel("Execution Time (microseconds)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(actual: List[float], predicted: List[float]):
    """Scatter of actual vs predicted execution time with an ideal y=x line."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual_arr, predicted_arr, color="#2f855a", alpha=0.7,
               label="Predictions")

    # Ideal line y = x: perfect predictions would lie exactly on this line.
    lo = float(min(actual_arr.min(), predicted_arr.min())) if len(actual_arr) else 0.0
    hi = float(max(actual_arr.max(), predicted_arr.max())) if len(actual_arr) else 1.0
    ax.plot([lo, hi], [lo, hi], color="#c53030", linestyle="--", label="Ideal (y = x)")

    ax.set_title("Actual vs AI-Predicted Execution Time")
    ax.set_xlabel("Actual Execution Time (microseconds)")
    ax.set_ylabel("Predicted Execution Time (microseconds)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_single_actual_vs_predicted(actual_us: float, predicted_us: float):
    """Simple two-bar comparison for the current single simulation run."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = ["Actual", "AI Predicted"]
    values = [actual_us, predicted_us]
    ax.bar(labels, values, color=["#2b6cb0", "#2f855a"])
    ax.set_title("Current Run: Actual vs Predicted Execution Time")
    ax.set_ylabel("Execution Time (microseconds)")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_hitrate_vs_execution_time(df: pd.DataFrame):
    """Scatter showing how cache hit rate relates to execution time."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["cache_hit_rate"], df["execution_time_us"],
               color="#805ad5", alpha=0.7)
    ax.set_title("Cache Hit Rate vs Execution Time")
    ax.set_xlabel("Cache Hit Rate (%)")
    ax.set_ylabel("Execution Time (microseconds)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_frequency_vs_execution_time(df: pd.DataFrame):
    """Scatter showing the inverse relationship between frequency and time."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["clock_frequency_ghz"], df["execution_time_us"],
               color="#dd6b20", alpha=0.7)
    ax.set_title("Clock Frequency vs Execution Time")
    ax.set_xlabel("Clock Frequency (GHz)")
    ax.set_ylabel("Execution Time (microseconds)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
