from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from telemetry_tool import load_telemetry, detect_anomalies
from telemetry_tool.data_loader import SENSOR_COLUMNS
from run_analysis import build_daily_averages


st.set_page_config(
    page_title="FutureFix Telemetry Monitor",
    layout="wide",
)

st.title("FutureFix Gear Line 3 — Telemetry Explorer")

st.caption(
    "Exploratory anomaly detection. "
    "A flag is a review prompt, not proof of machine failure."
)

data_path = st.sidebar.text_input(
    "Telemetry folder or CSV path",
    value="data",
)

contamination = st.sidebar.slider(
    "Expected anomaly fraction",
    min_value=0.005,
    max_value=0.10,
    value=0.01,
    step=0.005,
)

window = st.sidebar.selectbox(
    "Rolling window (readings)",
    options=[3, 5, 10, 20],
    index=1,
)


@st.cache_data
def load_cached(path):
    return load_telemetry(path)


try:
    data = load_cached(data_path)
except Exception as exc:
    st.error(f"Could not load telemetry: {exc}")
    st.info(
        "Put telemetry_*.csv files in the data folder "
        "or enter a valid CSV path."
    )
    st.stop()


@st.cache_data
def detect_cached(data, contamination, window):
    return detect_anomalies(
        data,
        contamination=contamination,
        window=window,
    )


with st.spinner("Running anomaly detection..."):
    results = detect_cached(
        data,
        contamination,
        window,
    )


machines = sorted(results["machine_id"].unique())

machine = st.sidebar.selectbox(
    "Machine",
    machines,
)

sensor = st.sidebar.selectbox(
    "Sensor",
    SENSOR_COLUMNS,
)

daily = build_daily_averages(data)
daily_machine = daily[daily["machine_id"] == machine]

machine_results = results[
    results["machine_id"] == machine
].copy()

alerts = machine_results[
    machine_results["is_anomaly"]
]

col1, col2, col3 = st.columns(3)

col1.metric("Total readings", f"{len(data):,}")
col2.metric("Selected machine alerts", f"{len(alerts):,}")
col3.metric("Machines", len(machines))


st.subheader(f"Daily average: {sensor}")

if not daily_machine.empty:
    fig = px.line(
        daily_machine,
        x="date",
        y=sensor,
        markers=True,
        labels={
            "date": "Date",
            sensor: f"Daily average ({sensor})",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


st.subheader("Daily averages table")

st.dataframe(
    daily_machine,
    use_container_width=True,
    height=220,
)


st.subheader("Raw telemetry")

only_anomalies = st.checkbox(
    "Show only model-flagged readings",
    value=False,
)

raw_view = machine_results

if only_anomalies:
    raw_view = raw_view[
        raw_view["is_anomaly"]
    ]

st.caption(
    f"Showing {len(raw_view):,} rows. "
    "Scroll vertically and horizontally to inspect readings."
)

st.dataframe(
    raw_view,
    use_container_width=True,
    height=450,
)


st.subheader("Flagged readings")

if alerts.empty:
    st.info(
        "No readings were flagged for this machine "
        "under the current model settings."
    )
else:
    st.dataframe(
        alerts.sort_values(
            "anomaly_score",
            ascending=False,
        ),
        use_container_width=True,
        height=350,
    )


st.download_button(
    label="Download selected machine data as CSV",
    data=raw_view.to_csv(index=False).encode("utf-8"),
    file_name=f"{machine}_telemetry.csv",
    mime="text/csv",
)