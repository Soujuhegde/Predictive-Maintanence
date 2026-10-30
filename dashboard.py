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

SENSOR_LABELS = {
    "spindle_speed_rpm": "Spindle speed",
    "spindle_load_pct": "Spindle load",
    "spindle_current_a": "Spindle current",
    "motor_temp_c": "Motor temperature",
    "vibration_mm_s": "Vibration",
    "coolant_flow_lpm": "Coolant flow",
}

SENSOR_UNITS = {
    "spindle_speed_rpm": "rpm",
    "spindle_load_pct": "%",
    "spindle_current_a": "A",
    "motor_temp_c": "°C",
    "vibration_mm_s": "mm/s",
    "coolant_flow_lpm": "L/min",
}

st.markdown(
    """
    <style>
    :root {
        --page: #f5f3ed;
        --surface: #fffefa;
        --ink: #26342f;
        --muted: #68756e;
        --line: #e4e1d8;
        --green: #567b67;
        --green-soft: #edf3ee;
        --amber: #a76e35;
    }

    .stApp {
        background: var(--page);
        color: var(--ink);
        font-family: Inter, "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #eeece5;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .block-container {
        padding-top: 2.4rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3, [data-testid="stMetricValue"] {
        color: var(--ink);
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: 2rem;
        font-weight: 650;
        margin-bottom: 0.25rem;
    }

    h2, h3 {
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1rem 1.15rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.82rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
        font-weight: 650;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: var(--surface);
        border-color: var(--line);
        border-radius: 10px;
    }

    [data-testid="stTabs"] button {
        color: var(--muted);
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--green);
    }

    [data-testid="stDownloadButton"] button {
        color: var(--green);
        border-color: #b7c8bb;
        background: var(--green-soft);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }

    hr {
        border-color: var(--line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gear Line 3")
st.caption(
    "Telemetry overview  /  Exploratory anomaly detection. "
    "A flag is a review prompt, not proof of machine failure."
)

default_data_path = Path(__file__).resolve().parent / "FutureFix_GearLine3_Dataset"

with st.sidebar:
    st.subheader("View settings")
    data_path = st.text_input(
        "Telemetry folder or CSV path",
        value=str(default_data_path),
    )
    contamination = st.slider(
        "Expected anomaly fraction",
        min_value=0.005,
        max_value=0.10,
        value=0.01,
        step=0.005,
        format="%.3f",
    )
    window = st.selectbox(
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
    results = detect_cached(data, contamination, window)


machines = sorted(results["machine_id"].unique())

with st.sidebar:
    machine = st.selectbox("Machine", machines)
    sensor = st.selectbox(
        "Trend sensor",
        SENSOR_COLUMNS,
        format_func=lambda column: SENSOR_LABELS[column],
    )

daily = build_daily_averages(data)
daily_machine = daily[daily["machine_id"] == machine]
machine_results = results[results["machine_id"] == machine].copy()
alerts = machine_results[machine_results["is_anomaly"]]
alert_rate = len(alerts) / len(machine_results) * 100 if len(machine_results) else 0

metric_columns = st.columns(3, gap="medium")
with metric_columns[0]:
    st.metric("Telemetry readings", f"{len(data):,}")
with metric_columns[1]:
    st.metric("Flagged on this machine", f"{len(alerts):,}")
with metric_columns[2]:
    st.metric("Machines monitored", len(machines))

st.write("")
chart_column, readout_column = st.columns([2.15, 1], gap="large")

with chart_column:
    with st.container(border=True):
        st.subheader(f"{SENSOR_LABELS[sensor]} trend")
        st.caption(
            f"Daily average · {SENSOR_UNITS[sensor]} · {machine}"
        )

        if not daily_machine.empty:
            fig = px.line(
                daily_machine,
                x="date",
                y=sensor,
                markers=True,
                labels={
                    "date": "Date",
                    sensor: f"{SENSOR_LABELS[sensor]} ({SENSOR_UNITS[sensor]})",
                },
            )
            fig.update_traces(
                line={"color": "#567b67", "width": 2.5},
                marker={
                    "color": "#fffefa",
                    "line": {"color": "#567b67", "width": 2},
                    "size": 8,
                },
                hovertemplate=(
                    "%{x|%b %d, %Y}<br>"
                    f"{SENSOR_LABELS[sensor]}: %{{y:.2f}} "
                    f"{SENSOR_UNITS[sensor]}<extra></extra>"
                ),
            )
            fig.update_layout(
                height=350,
                margin={"l": 8, "r": 12, "t": 12, "b": 8},
                paper_bgcolor="#fffefa",
                plot_bgcolor="#fffefa",
                font={"family": "Inter, Segoe UI, sans-serif", "color": "#52615a"},
                hoverlabel={
                    "bgcolor": "#26342f",
                    "font": {"color": "#fffefa"},
                },
                xaxis={
                    "showgrid": False,
                    "linecolor": "#e4e1d8",
                    "tickformat": "%b %d",
                    "tickfont": {"color": "#68756e"},
                },
                yaxis={
                    "title": None,
                    "gridcolor": "#eeece5",
                    "zeroline": False,
                    "tickfont": {"color": "#68756e"},
                },
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No daily readings are available for this machine.")

with readout_column:
    with st.container(border=True):
        st.subheader("Review")
        st.metric("Flagged readings", f"{len(alerts):,}")
        st.metric("Share of machine readings", f"{alert_rate:.2f}%")
        st.markdown("---")
        st.caption(
            "Flags are relative outliers from the selected model settings. "
            "Review them alongside operating context before drawing conclusions."
        )

st.write("")
daily_tab, readings_tab, flagged_tab = st.tabs(
    ["Daily averages", "Telemetry readings", "Flagged readings"]
)

with daily_tab:
    st.subheader("Daily averages")
    st.caption(f"Sensor averages for {machine}, grouped by calendar day.")
    st.dataframe(
        daily_machine,
        width="stretch",
        hide_index=True,
    )

with readings_tab:
    st.subheader("Telemetry readings")
    only_anomalies = st.checkbox(
        "Show only model-flagged readings",
        value=False,
    )
    raw_view = alerts if only_anomalies else machine_results
    st.caption(
        f"{len(raw_view):,} readings shown · Scroll to inspect all fields."
    )
    st.dataframe(
        raw_view,
        width="stretch",
        height=450,
        hide_index=True,
    )
    st.download_button(
        label="Download visible readings as CSV",
        data=raw_view.to_csv(index=False).encode("utf-8"),
        file_name=f"{machine}_telemetry.csv",
        mime="text/csv",
    )

with flagged_tab:
    st.subheader("Flagged readings")
    st.caption("Highest anomaly scores appear first.")
    if alerts.empty:
        st.info(
            "No readings were flagged for this machine "
            "under the current model settings."
        )
    else:
        st.dataframe(
            alerts.sort_values("anomaly_score", ascending=False),
            width="stretch",
            height=450,
            hide_index=True,
        )
