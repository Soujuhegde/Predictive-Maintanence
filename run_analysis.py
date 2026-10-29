import argparse
from pathlib import Path

import pandas as pd

from telemetry_tool import load_telemetry, detect_anomalies
from telemetry_tool.data_loader import SENSOR_COLUMNS


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "FutureFix_GearLine3_Dataset"


def build_daily_averages(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily average for each machine and sensor."""
    work = data.copy()
    work["date"] = work["ts"].dt.date

    return (
        work.groupby(
            ["machine_id", "date"],
            as_index=False,
        )[SENSOR_COLUMNS]
        .mean()
        .sort_values(["machine_id", "date"])
    )


def main():
    parser = argparse.ArgumentParser(
        description="FutureFix telemetry anomaly analysis"
    )

    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Telemetry CSV file or directory",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for output CSV files",
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Expected outlier fraction",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling feature window",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_telemetry(args.data_dir)

    results = detect_anomalies(
        data,
        contamination=args.contamination,
        window=args.window,
    )

    results.to_csv(
        output_dir / "anomaly_flags.csv",
        index=False,
    )

    results[results["is_anomaly"]].to_csv(
        output_dir / "anomalies_only.csv",
        index=False,
    )

    build_daily_averages(data).to_csv(
        output_dir / "daily_averages.csv",
        index=False,
    )

    summary = (
        results.groupby("machine_id")
        .agg(
            readings=("is_anomaly", "size"),
            anomalies=("is_anomaly", "sum"),
        )
        .reset_index()
    )

    summary["anomaly_percent"] = (
        summary["anomalies"]
        / summary["readings"]
        * 100
    ).round(3)

    summary.to_csv(
        output_dir / "model_summary.csv",
        index=False,
    )

    print(f"Loaded {len(data):,} readings.")
    print("\nAnomaly summary:")
    print(summary.to_string(index=False))
    print(f"\nOutput folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()