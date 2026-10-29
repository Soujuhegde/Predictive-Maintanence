
from pathlib import Path
import pandas as pd
from ydata_profiling import ProfileReport

# Folder where this Python script is saved
BASE_DIR = Path(__file__).resolve().parent

# Output folder for all reports
OUTPUT_DIR = BASE_DIR / "profiling_reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset files are stored in the project's dataset folder.
DATA_DIR = BASE_DIR / "FutureFix_GearLine3_Dataset"

# Find all CSV files directly in the dataset folder
csv_files = sorted(
    file for file in DATA_DIR.glob("*.csv")
)

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}. "
        "Place the dataset CSV files in that folder."
    )

print(f"Found {len(csv_files)} CSV files.")

inventory = []

for number, csv_path in enumerate(csv_files, start=1):
    print(f"\n[{number}/{len(csv_files)}] Processing {csv_path.name}")

    try:
        # Read the original CSV
        df = pd.read_csv(csv_path)

        # Use a copy for profiling
        profile_df = df.copy()

        # Treat -999 as missing only for the current
        # telemetry profile. Keep the original CSV unchanged.
        if (
            csv_path.name.startswith("telemetry_")
            and "spindle_current_a" in profile_df.columns
        ):
            profile_df["spindle_current_a"] = (
                profile_df["spindle_current_a"].replace(-999, float("nan"))
            )

        # Generate the profiling report
        report = ProfileReport(
            profile_df,
            title=f"Dataset Profile - {csv_path.name}",
            explorative=True
        )

        # Save the report as HTML
        report_path = OUTPUT_DIR / f"{csv_path.stem}_profile.html"
        report.to_file(str(report_path))

        # Collect basic dataset information
        inventory.append({
            "file_name": csv_path.name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "report_file": report_path.name,
            "status": "Success"
        })

        print(f"Report saved: {report_path.name}")

    except Exception as error:
        print(f"Error processing {csv_path.name}: {error}")

        inventory.append({
            "file_name": csv_path.name,
            "rows": None,
            "columns": None,
            "missing_cells": None,
            "duplicate_rows": None,
            "report_file": "",
            "status": f"Failed: {error}"
        })

# Save an inventory of all datasets
inventory_df = pd.DataFrame(inventory)
inventory_path = OUTPUT_DIR / "dataset_inventory.csv"
inventory_df.to_csv(inventory_path, index=False)

successful = sum(item["status"] == "Success" for item in inventory)

print("\n========== COMPLETE ==========")
print(f"CSV files found: {len(csv_files)}")
print(f"Reports generated: {successful}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Inventory: {inventory_path}")
