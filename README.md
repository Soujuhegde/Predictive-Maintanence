# FutureFix Machine Telemetry Anomaly Tool

A Python tool for exploring the supplied Gear Line 3 telemetry, detecting
unusual sensor patterns, reviewing daily averages, and inspecting raw data.
It includes batch analysis, a Streamlit dashboard, and a queue-based IoT
replay.

## Dataset

The supplied telemetry is in `FutureFix_GearLine3_Dataset/`:

- `telemetry_HOB-01.csv`
- `telemetry_HOB-02.csv`
- `telemetry_SHV-01.csv`

Each file contains timestamps, a machine ID, spindle speed/load/current,
motor temperature, vibration, and coolant flow. These files contain three
machines and do not have the assignment brief's separate `Hob_1` /
`Hob_2` files or axis-specific X/Z/V current and temperature columns. This
analysis uses the dataset present in this repository; its signals should not
be described as axis-specific measurements.

`spindle_current_a == -999` is treated as missing because it is a special
value in the supplied files; its meaning should be confirmed with the data
owner. Machine-state intervals are available in `machine_states.csv` and are
used below to contextualize results, not as detector inputs.

## Setup

Use Python 3.10 or later. From the repository root:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt

## Run analysis

The checked-in dataset is the default:

    python run_analysis.py

To choose paths or model settings:

    python run_analysis.py --data-dir FutureFix_GearLine3_Dataset --output-dir outputs --contamination 0.01 --window 5

The script writes model flags/scores, flagged rows, daily averages, and a
per-machine summary to the output directory.

## Generate profiling reports

The HTML profiling reports are generated artifacts and are excluded from
version control so they do not dominate GitHub's language statistics.
Generate them locally when needed:

    python generate_profiles.py

## Findings from the supplied data

The following results are from running the Isolation Forest batch analysis
on all 241,490 telemetry rows with `contamination=0.01`, `window=5`, and
`random_state=42`:

| Machine | Readings | Model flags | Flagged |
|---|---:|---:|---:|
| HOB-01 | 80,016 | 801 | 1.001% |
| HOB-02 | 80,737 | 808 | 1.001% |
| SHV-01 | 80,737 | 808 | 1.001% |

The independent global IQR check in `anomaly_results/iqr_anomaly_summary.csv`
reported zero outliers for all 18 machine/sensor combinations. That is not
evidence that the machines are healthy: the IQR bounds are calculated across
idle, running, setup, and stopped readings, which can make broad thresholds
insensitive to unusual behavior within one operating mode.

Re-run this separate IQR check with:

    python detect_anomalies.py

Machine-state intervals label more than 99.7% of telemetry rows and show why
the Isolation Forest flags also need context. The model was fit per machine,
but not per operating state; the contamination setting also requests roughly
1% of each machine's readings to be flagged. A post-run timestamp join gives:

| Machine | Flag rate while RUNNING | Flag rate while DOWN |
|---|---:|---:|
| HOB-01 | 1.03% | 4.14% |
| HOB-02 | 0.96% | 13.56% |
| SHV-01 | 0.96% | 3.18% |

In particular, HOB-02 has 375 of its 808 flags during intervals labelled
`DOWN`. This may reflect state transitions or a mismatch between coarse
state intervals and 15-second telemetry; it is not, by itself, evidence of
a fault. The model also flags readings during `RUNNING`, but without verified
engineering limits or confirmed failure/maintenance labels those are
investigation candidates, not diagnoses.

**Conclusion: the available evidence is inconclusive, not a clean bill of
health and not a confirmed fault.** The IQR check found no global extremes;
the Isolation Forest found relative outliers under a threshold that controls
the number of flags. HOB-02's state-conditioned flag concentration merits
review with the plant engineer, alongside the state timeline and operating
events. A stronger next analysis would validate state alignment, compare
like-for-like running periods (and production conditions), and evaluate
state-specific baselines against engineering limits and maintenance outcomes.

## Run dashboard

    streamlit run dashboard.py

The dashboard uses an off-white operations view with a daily sensor trend,
machine-level review summary, and separate tabs for daily averages, telemetry
readings, and flagged readings. The selected sensor, machine, and model settings
can be changed in the sidebar; visible telemetry readings can be downloaded as
CSV.

## Simulate IoT replay

    python simulate_iot.py --delay 0

The replay splits each machine's rows chronologically, fits on the earlier
portion, and queues later rows through the streaming detector. It writes
scores and flags to `outputs/stream_results.csv`. This is a local simulation,
not a live sensor connection or production alarm.

## Run tests

    python -m pytest -q

## Python interface

    from telemetry_tool import load_telemetry, detect_anomalies

    data = load_telemetry("FutureFix_GearLine3_Dataset")
    results = detect_anomalies(data, contamination=0.01, window=5)
    alerts = results[results["is_anomaly"]]

## Detection approach and limitations

Isolation Forest is fit independently for each machine using sensor values
and short-window rolling means, standard deviations, and changes. It detects
relative unusualness, not a known fault class. The stream detector applies
the same per-machine idea to incoming rows using a historical baseline.

- The contamination value is a model threshold, not an estimated failure
  rate; changing it changes the number of flags.
- Batch models pool operating states, and the detector does not yet use
  machine state, production events, verified engineering limits, or
  maintenance labels.
- Missing feature values are imputed for modelling; duplicate timestamps
  and irregular data should be reviewed before operational use.
- Results are exploratory. A flag is not proof of equipment failure, and
  the model is not a validated real-time alarm system.
- Review results with a plant engineer before using them for maintenance
  decisions.

## AI tooling

GitHub Copilot (Copilot SDK in VS Code) was used to review and edit the
project documentation and code, and to help run tests and data analysis.
The analysis settings and reported results were checked against a local run
on the supplied dataset.