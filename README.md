# FutureFix Machine Telemetry Anomaly Tool

A Python tool for exploring machine telemetry, detecting unusual
sensor patterns, reviewing daily averages, and inspecting raw data.

## Features

- Loads one telemetry CSV or multiple telemetry CSVs.
- Uses scikit-learn Isolation Forest for unsupervised anomaly detection.
- Builds rolling mean, rolling standard deviation, and change features.
- Fits a separate model for each machine.
- Exports anomaly scores, flagged rows, daily averages, and summaries.
- Provides a Streamlit dashboard with interactive charts and raw-data
  browsing.
- Includes unit tests using synthetic telemetry.

## Dataset

The supplied telemetry contains:

- spindle_speed_rpm
- spindle_load_pct
- spindle_current_a
- motor_temp_c
- vibration_mm_s
- coolant_flow_lpm

The assignment mentions axis-specific current and temperature channels,
but these are not present in the supplied telemetry files.

The value -999 in spindle_current_a is treated as missing. Its meaning
should be confirmed with the data owner.

## Setup

Install Python 3.10 or later, then run:

    python -m venv .venv

On Windows:

    .venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

Place the telemetry CSV files in the data/ directory.

## Run analysis

    python run_analysis.py --data-dir data --output-dir outputs

Optional settings:

    python run_analysis.py --data-dir data --output-dir outputs --contamination 0.01 --window 5

The contamination parameter is the model's expected outlier fraction.
It is a tuning parameter, not the actual machine fault rate.

## Run dashboard

    streamlit run dashboard.py

Select a machine and sensor in the sidebar. The dashboard displays
daily averages, a scrollable raw-data table, anomaly flags, and a
CSV download option.

## Run tests

    pytest -q

## Python interface

    from telemetry_tool import load_telemetry, detect_anomalies

    data = load_telemetry("data")
    results = detect_anomalies(data, contamination=0.01, window=5)
    alerts = results[results["is_anomaly"]]

## Detection approach

Isolation Forest is an unsupervised ML algorithm that identifies
observations that are relatively easy to isolate from the rest of
the dataset. Sensor levels and short-term rolling features are used
to capture unusual combinations and changes.

Models are fitted separately per machine because the machines may
have different operating characteristics.

## Limitations

- This is an exploratory batch detector, not a validated real-time
  alarm system.
- A flagged reading is not proof of equipment failure.
- The contamination parameter affects how many readings are flagged.
- Machine state, production events, maintenance records, and verified
  engineering limits are not yet included.
- Missing feature values are imputed for modelling.
- Results must be reviewed with a plant engineer before operational use.