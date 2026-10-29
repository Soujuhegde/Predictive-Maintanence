from pathlib import Path

import pandas as pd


SENSOR_COLUMNS = [
    "spindle_speed_rpm",
    "spindle_load_pct",
    "spindle_current_a",
    "motor_temp_c",
    "vibration_mm_s",
    "coolant_flow_lpm",
]

REQUIRED_COLUMNS = {
    "ts",
    "machine_id",
    *SENSOR_COLUMNS,
}


def load_telemetry(source: str | Path) -> pd.DataFrame:
    """Load and clean one telemetry CSV or all telemetry CSVs in a folder."""
    path = Path(source)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("telemetry_*.csv"))
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    if not files:
        raise FileNotFoundError(
            f"No telemetry_*.csv files found in {path}"
        )

    frames = []

    for file in files:
        frame = pd.read_csv(file)

        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f"{file.name} is missing columns: {sorted(missing)}"
            )

        frame["source_file"] = file.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    data["ts"] = pd.to_datetime(data["ts"], errors="coerce")
    data = data.dropna(subset=["ts", "machine_id"]).copy()

    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    # The supplied data uses -999 as a special current value.
    # Treat it as missing until its meaning is confirmed.
    data.loc[
        data["spindle_current_a"] == -999,
        "spindle_current_a",
    ] = float("nan")

    return (
        data.sort_values(["machine_id", "ts"])
        .reset_index(drop=True)
    )