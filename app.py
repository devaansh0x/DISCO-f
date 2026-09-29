"""
DISCO - Dynamic Instruction Simulation for Computer Organization
app.py  (Streamlit front end)

This is the main application. It ties together every DISCO component so a
professor can run one program and follow the whole story:

    input workload + CPU config
        -> CPU simulation (fetch/decode/execute/mem/writeback)
        -> COA performance metrics (cycles, CPI, cache hit rate, exec time)
        -> ML prediction of execution time
        -> actual vs predicted comparison
        -> tables and graphs

Run it (from inside the DISCO folder) with:

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

# Make sure the DISCO package folder is importable when Streamlit runs the file
# directly (so `simulator`, `analysis`, etc. resolve as top-level packages).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from ai.predictor import (
    ExecutionTimePredictor,
    PredictorError,
    prediction_error,
)
from analysis.dataset import (
    DATASET_PATH,
    generate_and_save,
    load_dataset,
)
from analysis.metrics import MetricsError, compute_metrics, metrics_table
from simulator.cache import Cache
from simulator.cpu import CPU, CPUError
from simulator.instructions import InstructionError, parse_program, supported_opcodes
from simulator.memory import Memory
from visualization.plots import (
    plot_actual_vs_predicted,
    plot_frequency_vs_execution_time,
    plot_hitrate_vs_execution_time,
    plot_single_actual_vs_predicted,
    plot_workload_vs_execution_time,
)

EXAMPLE_WORKLOAD = """LOAD R1, 10
LOAD R2, 20
ADD R3, R1, R2
MUL R4, R3, R2
STORE R4, 100"""


# ---------------------------------------------------------------------------
# Dataset / model caching helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _ensure_dataset(num_samples: int = 150) -> pd.DataFrame:
    """Load the dataset, generating it from the real simulator if missing."""
    if not os.path.exists(DATASET_PATH) or os.path.getsize(DATASET_PATH) == 0:
        return generate_and_save(num_samples=num_samples)
    return load_dataset()


@st.cache_resource(show_spinner=False)
def _train_predictor(dataset_signature: int):
    """Train the ML model once and cache it. Signature busts the cache."""
    predictor = ExecutionTimePredictor()
    df = load_dataset()
    result = predictor.train(df)
    return predictor, result, df


def main() -> None:
    st.set_page_config(page_title="DISCO", page_icon=None, layout="wide")

    st.title("DISCO")
    st.subheader("Dynamic Instruction Simulation for Computer Organization")
    st.caption(
        "An AI-based dynamic instruction simulation and performance "
        "prediction system for Computer Organization."
    )

    # ------------------------------------------------------------------
    # Sidebar: workload + CPU configuration
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("1. Workload")
        st.write("Enter a small instruction sequence.")
        st.caption("Supported: " + ", ".join(supported_opcodes()))
        workload = st.text_area("Instructions", value=EXAMPLE_WORKLOAD, height=180)
        if st.button("Load example workload"):
            workload = EXAMPLE_WORKLOAD
            st.rerun()

        st.header("2. CPU Configuration")
        clock_frequency_ghz = st.number_input(
            "Clock Frequency (GHz)", min_value=0.1, max_value=10.0,
            value=2.0, step=0.1,
        )
        num_cache_lines = st.select_slider(
            "Cache Lines (direct-mapped)", options=[2, 4, 8, 16, 32, 64], value=8
        )
        num_registers = st.slider("Registers (R0..Rn-1)", min_value=4,
                                  max_value=32, value=8, step=1)
        mem_size = st.number_input("Memory Size (words)", min_value=16,
                                   max_value=4096, value=256, step=16)

        st.header("3. Run")
        run_clicked = st.button("Run Simulation", type="primary")

        with st.expander("Dataset / Model settings"):
            regen = st.button("Regenerate dataset from simulator")
            if regen:
                _ensure_dataset.clear()
                _train_predictor.clear()
                generate_and_save(num_samples=150)
                st.success("Dataset regenerated from the simulator.")

    # Show the instruction cost reference so the COA assumptions are explicit.
    with st.expander("Simulation assumptions (instruction cycle costs)"):
        st.write(
            "These are educational SIMULATION ASSUMPTIONS, not exact timings "
            "of a real commercial CPU."
        )
        st.table(
            pd.DataFrame(
                [
                    {"Instruction": "LOAD", "Cycles": 2, "Type": "Memory"},
                    {"Instruction": "STORE", "Cycles": 2, "Type": "Memory"},
                    {"Instruction": "ADD", "Cycles": 1, "Type": "ALU"},
                    {"Instruction": "SUB", "Cycles": 1, "Type": "ALU"},
                    {"Instruction": "MUL", "Cycles": 2, "Type": "ALU"},
                    {"Instruction": "MOV", "Cycles": 1, "Type": "Register"},
                ]
            )
        )

    if not run_clicked:
        st.info("Configure a workload and CPU settings in the sidebar, then "
                "click **Run Simulation**.")
        _show_dataset_overview()
        return

    # ------------------------------------------------------------------
    # Step 1: parse + simulate
    # ------------------------------------------------------------------
    try:
        program = parse_program(workload)
    except InstructionError as exc:
        st.error(f"Workload error: {exc}")
        return

    cpu = CPU(
        num_registers=int(num_registers),
        memory=Memory(size=int(mem_size)),
        cache=Cache(num_lines=int(num_cache_lines)),
    )
    try:
        cpu.run(program)
    except (CPUError, Exception) as exc:  # surface runtime issues cleanly
        st.error(f"Simulation error: {exc}")
        return

    # ------------------------------------------------------------------
    # Step 2: metrics
    # ------------------------------------------------------------------
    try:
        metrics = compute_metrics(cpu, float(clock_frequency_ghz))
    except MetricsError as exc:
        st.error(f"Metrics error: {exc}")
        return

    # ------------------------------------------------------------------
    # Layout: results
    # ------------------------------------------------------------------
    st.success("Simulation complete.")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("### Execution Trace")
        st.caption("Fetch -> Decode -> Execute -> Memory -> Write Back per instruction.")
        st.dataframe(pd.DataFrame(cpu.trace_as_dicts()), use_container_width=True)

    with col_b:
        st.markdown("### Registers (final state)")
        reg_df = pd.DataFrame(
            [{"Register": k, "Value": v} for k, v in cpu.register_snapshot().items()]
        )
        st.dataframe(reg_df, use_container_width=True, height=300)
        st.markdown(f"**Program Counter (final):** {cpu.pc}")

    # ------------------------------------------------------------------
    # Step 3: ML prediction
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## COA Performance Metrics & AI Prediction")

    predictor = None
    training_result = None
    dataset = None
    predicted_us = None
    err = None
    ml_error_msg = None

    try:
        dataset = _ensure_dataset()
        predictor, training_result, dataset = _train_predictor(len(dataset))
        predicted_us = predictor.predict_one(metrics)
        err = prediction_error(metrics["execution_time_us"], predicted_us)
    except (PredictorError, FileNotFoundError) as exc:
        ml_error_msg = str(exc)

    # Performance + prediction summary table.
    table = metrics_table(metrics)
    if predicted_us is not None:
        table["AI Predicted Execution Time (microseconds)"] = f"{predicted_us:.6f}"
        table["Prediction Absolute Error (microseconds)"] = f"{err['absolute_error_us']:.6f}"
        table["Prediction Error (%)"] = f"{err['percentage_error']:.2f}"

    summary_df = pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in table.items()]
    )

    m_col, g_col = st.columns([2, 3])
    with m_col:
        st.markdown("### Results Table")
        st.dataframe(summary_df, use_container_width=True, height=520)

    with g_col:
        st.markdown("### Current Run: Actual vs Predicted")
        if predicted_us is not None:
            st.pyplot(plot_single_actual_vs_predicted(
                metrics["execution_time_us"], predicted_us))
        elif ml_error_msg:
            st.warning(f"ML prediction unavailable: {ml_error_msg}")

    if ml_error_msg is None and training_result is not None:
        st.markdown("### ML Model (Linear Regression)")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Test MAE (microseconds)", f"{training_result.mae:.6f}")
        mc2.metric("Test R^2", f"{training_result.r2:.4f}")
        mc3.metric("Train / Test rows", f"{training_result.n_train} / {training_result.n_test}")
        with st.expander("Learned model parameters"):
            coeff_df = pd.DataFrame(
                [{"Feature": k, "Coefficient": v}
                 for k, v in training_result.coefficients.items()]
                + [{"Feature": "intercept", "Coefficient": training_result.intercept}]
            )
            st.table(coeff_df)
            st.caption(
                "Execution time is predicted from: instruction count, CPI, "
                "cache hit rate and clock frequency."
            )

    # ------------------------------------------------------------------
    # Step 4: dataset-wide graphs
    # ------------------------------------------------------------------
    if dataset is not None:
        st.markdown("---")
        st.markdown("## Dataset Graphs")
        st.caption(
            f"Graphs built from {len(dataset)} real simulator runs stored in "
            f"data/simulation_data.csv."
        )

        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(plot_workload_vs_execution_time(dataset))
            st.pyplot(plot_hitrate_vs_execution_time(dataset))
        with g2:
            st.pyplot(plot_frequency_vs_execution_time(dataset))
            if predictor is not None:
                actual = dataset["execution_time_us"].tolist()
                predicted = [
                    predictor.predict_one(row)
                    for row in dataset.to_dict(orient="records")
                ]
                st.pyplot(plot_actual_vs_predicted(actual, predicted))


def _show_dataset_overview() -> None:
    """Show a small preview of the dataset on the landing screen."""
    try:
        df = _ensure_dataset()
    except Exception as exc:  # dataset generation problem
        st.warning(f"Could not prepare dataset yet: {exc}")
        return
    st.markdown("### Simulation Dataset Preview")
    st.caption(
        f"{len(df)} rows generated by actually running the DISCO simulator. "
        "Stored in data/simulation_data.csv."
    )
    st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
