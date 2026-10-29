
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "FutureFix_GearLine3_Dataset"
OUTPUT_DIR = BASE_DIR / "anomaly_results"
OUTPUT_DIR.mkdir(exist_ok=True)

machines = ["HOB-01", "HOB-02", "SHV-01"]

sensors = [
    "spindle_speed_rpm",
    "spindle_load_pct",
    "spindle_current_a",
    "motor_temp_c",
    "vibration_mm_s",
    "coolant_flow_lpm",
]

all_flags = []
summary = []

for machine in machines:
    path = DATA_DIR / f"telemetry_{machine}.csv"

    if not path.exists():
        print(f"File not found: {path.name}")
        continue

    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # Preserve all original columns in the output.
    df["machine_id"] = machine

    machine_flags = []

    for sensor in sensors:
        if sensor not in df.columns:
            continue

        values = pd.to_numeric(df[sensor], errors="coerce")

        # -999 is treated as an invalid/special code for
        # current, not as a real physical measurement.
        if sensor == "spindle_current_a":
            values = values.mask(values == -999)

        valid = values.dropna()

        if len(valid) < 4:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flagged = ((values < lower) | (values > upper)).fillna(False)

        count = int(flagged.sum())

        summary.append({
            "machine": machine,
            "sensor": sensor,
            "valid_readings": len(valid),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_threshold": lower,
            "upper_threshold": upper,
            "flagged_readings": count,
            "flagged_percent": round(100 * count / len(valid), 3),
        })

        if count:
            rows = df.loc[flagged].copy()
            rows["sensor_flagged"] = sensor
            rows["sensor_value"] = values.loc[flagged]
            rows["lower_threshold"] = lower
            rows["upper_threshold"] = upper
            machine_flags.append(rows)

    if machine_flags:
        all_flags.extend(machine_flags)

    # Separate data-quality checks
    current_invalid = (
        (df["spindle_current_a"] == -999).sum()
        if "spindle_current_a" in df.columns else 0
    )

    print(f"\n{machine}")
    print("Rows:", len(df))
    print("Exact duplicate rows:", int(df.duplicated().sum()))
    print("Invalid current codes (-999):", int(current_invalid))
    print("Missing cells:", int(df.isna().sum().sum()))

# Save the sensor-level results
summary_df = pd.DataFrame(summary)
summary_path = OUTPUT_DIR / "iqr_anomaly_summary.csv"
summary_df.to_csv(summary_path, index=False)

# Save all flagged sensor readings
if all_flags:
    flagged_df = pd.concat(all_flags, ignore_index=True)
    flagged_path = OUTPUT_DIR / "iqr_flagged_readings.csv"
    flagged_df.to_csv(flagged_path, index=False)
    print("\nFlagged readings saved:", flagged_path)
else:
    print("\nNo IQR outliers were detected.")

print("Summary saved:", summary_path)
print("\nFlag counts by machine and sensor:")
print(
    summary_df[
        ["machine", "sensor", "flagged_readings", "flagged_percent"]
    ].to_string(index=False)
)
