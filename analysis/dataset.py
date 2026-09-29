"""
DISCO - Dynamic Instruction Simulation for Computer Organization
analysis/dataset.py

Generates and manages the simulation dataset used to train the ML model.

Every row in the dataset is produced by ACTUALLY RUNNING the DISCO CPU
simulator on a randomly generated (but valid) workload with a randomly
chosen CPU configuration. The measured COA metrics become the dataset
features and target. Nothing here is fabricated: the numbers come from
the simulator itself.

The dataset is stored at ``data/simulation_data.csv`` and can be loaded
back for ML training in ai/predictor.py.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

import pandas as pd

from analysis.metrics import compute_metrics
from simulator.cache import Cache
from simulator.cpu import CPU
from simulator.instructions import Instruction, INSTRUCTION_CYCLES
from simulator.memory import Memory

# Path to the CSV dataset (kept inside the locked data/ folder).
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASET_PATH = os.path.join(_DATA_DIR, "simulation_data.csv")

# Columns saved to CSV. The first four (plus frequency) are ML features and
# ``execution_time_us`` is the ML target.
DATASET_COLUMNS = [
    "instruction_count",
    "total_cycles",
    "cpi",
    "cache_hit_rate",
    "cache_misses",
    "clock_frequency_ghz",
    "execution_time_us",
]

_OPCODES = list(INSTRUCTION_CYCLES.keys())


def _random_instruction(rng: random.Random, num_registers: int,
                        address_pool: List[int]) -> Instruction:
    """Build one valid random instruction using the simulator's own parser rules.

    Memory addresses are drawn from a small ``address_pool`` (a working set)
    rather than the whole memory. Reusing a limited set of addresses is what
    makes the cache produce a realistic mix of hits and misses, so the dataset
    actually demonstrates the cache-behaviour vs performance relationship.
    """
    op = rng.choice(_OPCODES)
    reg = lambda: f"R{rng.randrange(num_registers)}"

    if op in ("LOAD", "STORE"):
        operands = [reg(), str(rng.choice(address_pool))]
    elif op in ("ADD", "SUB", "MUL"):
        operands = [reg(), reg(), reg()]
    else:  # MOV
        operands = [reg(), str(rng.randrange(0, 100))]

    raw = f"{op} " + ", ".join(operands)
    return Instruction(opcode=op, operands=operands, cycles=INSTRUCTION_CYCLES[op], raw=raw)


def _random_program(rng: random.Random, length: int, num_registers: int,
                    address_pool: List[int]) -> List[Instruction]:
    """Generate a random program of the given length over a fixed address pool."""
    return [_random_instruction(rng, num_registers, address_pool) for _ in range(length)]


def generate_samples(num_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a dataset by running the real simulator many times.

    Each sample uses a random workload length, cache size and clock
    frequency, then records the measured COA metrics.

    Args:
        num_samples: Number of simulation runs (rows) to generate.
        seed: Random seed for reproducibility (deterministic dataset).

    Returns:
        A pandas DataFrame with columns ``DATASET_COLUMNS``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = random.Random(seed)
    rows: List[Dict[str, float]] = []

    for _ in range(num_samples):
        program_length = rng.randint(5, 40)
        num_registers = 8
        mem_size = 256
        num_cache_lines = rng.choice([4, 8, 16, 32])
        clock_frequency_ghz = round(rng.uniform(1.0, 4.0), 2)

        # A limited working set of addresses so that LOAD/STORE reuse memory
        # locations. This produces a realistic spread of cache hit rates across
        # the dataset (cache size vs working-set size drives hits/misses).
        working_set = rng.randint(4, 24)
        address_pool = rng.sample(range(mem_size), k=min(working_set, mem_size))

        cpu = CPU(
            num_registers=num_registers,
            memory=Memory(size=mem_size),
            cache=Cache(num_lines=num_cache_lines),
        )
        program = _random_program(rng, program_length, num_registers, address_pool)
        cpu.run(program)

        metrics = compute_metrics(cpu, clock_frequency_ghz)
        rows.append({col: metrics[col] for col in DATASET_COLUMNS})

    return pd.DataFrame(rows, columns=DATASET_COLUMNS)


def save_dataset(df: pd.DataFrame, path: str = DATASET_PATH) -> str:
    """Save a dataset DataFrame to CSV, creating the folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the dataset CSV. Raises FileNotFoundError if it does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it first with generate_and_save()."
        )
    return pd.read_csv(path)


def generate_and_save(num_samples: int = 100, seed: int = 42,
                      path: str = DATASET_PATH) -> pd.DataFrame:
    """Generate a fresh dataset from the simulator and save it to CSV."""
    df = generate_samples(num_samples=num_samples, seed=seed)
    save_dataset(df, path)
    return df


def append_run(metrics: Dict[str, float], path: str = DATASET_PATH) -> None:
    """Append a single real simulation run (e.g. from the UI) to the dataset."""
    row = {col: metrics[col] for col in DATASET_COLUMNS}
    df = pd.DataFrame([row], columns=DATASET_COLUMNS)
    header_needed = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode="a", header=header_needed, index=False)


if __name__ == "__main__":
    # Allow generating the dataset directly: python -m analysis.dataset
    frame = generate_and_save()
    print(f"Generated {len(frame)} simulation rows -> {DATASET_PATH}")
