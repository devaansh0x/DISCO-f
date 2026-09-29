"""
DISCO - Dynamic Instruction Simulation for Computer Organization
analysis/metrics.py

Computes the core Computer Organization & Architecture (COA) performance
metrics from a completed CPU simulation. These metrics are the bridge
between the simulator and the ML predictor: the same numbers that describe
CPU performance also become the ML features/target.

Equations used (standard COA formulas):

    CPI            = Total Clock Cycles / Total Instructions
    Cache Hit Rate = (Hits / (Hits + Misses)) * 100
    Execution Time = Total Clock Cycles / Clock Frequency

Unit handling: clock frequency is provided in GHz. 1 GHz = 1e9 cycles/second,
so Execution Time (seconds) = Total Clock Cycles / (frequency_ghz * 1e9).
Results are also reported in microseconds for readability.
"""

from __future__ import annotations

from typing import Dict

from simulator.cpu import CPU


class MetricsError(ValueError):
    """Raised for invalid metric inputs (e.g. non-positive frequency)."""


def compute_metrics(cpu: CPU, clock_frequency_ghz: float) -> Dict[str, float]:
    """Compute COA performance metrics from a finished simulation.

    Args:
        cpu: A CPU instance that has already executed a program.
        clock_frequency_ghz: Clock frequency in GHz (must be > 0).

    Returns:
        A structured dict of metrics suitable for display, dataset rows,
        and the ML predictor.

    Raises:
        MetricsError: if frequency is not positive or nothing was executed.
    """
    if clock_frequency_ghz <= 0:
        raise MetricsError("Clock frequency must be a positive number (GHz).")

    instruction_count = cpu.instruction_count
    total_cycles = cpu.clock_cycles

    if instruction_count == 0:
        raise MetricsError("No instructions were executed; cannot compute metrics.")

    # CPI = cycles per instruction.
    cpi = total_cycles / instruction_count

    # Cache statistics come straight from the cache model.
    cache_stats = cpu.cache.stats()

    # Execution time. Frequency in GHz -> cycles per second = ghz * 1e9.
    frequency_hz = clock_frequency_ghz * 1e9
    execution_time_s = total_cycles / frequency_hz
    execution_time_us = execution_time_s * 1e6  # microseconds for readability

    return {
        "instruction_count": instruction_count,
        "total_cycles": total_cycles,
        "cpi": round(cpi, 4),
        "cache_accesses": cache_stats["cache_accesses"],
        "cache_hits": cache_stats["cache_hits"],
        "cache_misses": cache_stats["cache_misses"],
        "cache_hit_rate": cache_stats["cache_hit_rate"],
        "clock_frequency_ghz": clock_frequency_ghz,
        "execution_time_s": execution_time_s,
        "execution_time_us": round(execution_time_us, 6),
    }


def metrics_table(metrics: Dict[str, float]) -> Dict[str, str]:
    """Format metrics into a human-readable label -> value mapping for a table."""
    return {
        "Instructions Executed": f"{int(metrics['instruction_count'])}",
        "Total Clock Cycles": f"{int(metrics['total_cycles'])}",
        "CPI (Cycles Per Instruction)": f"{metrics['cpi']:.4f}",
        "Cache Accesses": f"{int(metrics['cache_accesses'])}",
        "Cache Hits": f"{int(metrics['cache_hits'])}",
        "Cache Misses": f"{int(metrics['cache_misses'])}",
        "Cache Hit Rate (%)": f"{metrics['cache_hit_rate']:.2f}",
        "Clock Frequency (GHz)": f"{metrics['clock_frequency_ghz']:.3f}",
        "Execution Time (microseconds)": f"{metrics['execution_time_us']:.6f}",
    }
