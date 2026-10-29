import argparse
import json
import queue
import threading
import time
from pathlib import Path

import pandas as pd

from telemetry_tool import load_telemetry
from telemetry_tool.streaming import StreamingAnomalyDetector


STOP = object()
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "FutureFix_GearLine3_Dataset"


def split_by_machine(data, train_fraction=0.70):
    """Chronologically split each machine into history and replay rows."""
    historical = []
    incoming = []

    for _, machine in data.groupby("machine_id"):
        machine = machine.sort_values("ts")
        split_at = int(len(machine) * train_fraction)

        historical.append(machine.iloc[:split_at])
        incoming.append(machine.iloc[split_at:])

    train = pd.concat(historical).sort_values("ts")
    stream = pd.concat(incoming).sort_values("ts")

    return train.reset_index(drop=True), stream.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Replay telemetry as simulated IoT events"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default="outputs/stream_results.csv")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds between simulated sensor messages; 0 means as fast as possible",
    )
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay cannot be negative")

    data = load_telemetry(args.data_dir)
    historical, incoming = split_by_machine(data)

    detector = StreamingAnomalyDetector(
        contamination=args.contamination,
        window=args.window,
    ).fit(historical)

    events = queue.Queue(maxsize=1000)
    results = []

    def sensor_producer():
        """Pretend each CSV row is a message from an IoT sensor."""
        try:
            for _, row in incoming.iterrows():
                events.put(row.to_dict())
                if args.delay:
                    time.sleep(args.delay)
        finally:
            events.put(STOP)

    producer = threading.Thread(
        target=sensor_producer,
        daemon=True,
    )
    producer.start()

    print(f"Historical rows used for training: {len(historical):,}")
    print(f"Incoming rows to replay: {len(incoming):,}")
    print("Starting simulated IoT stream...\n")

    while True:
        event = events.get()

        if event is STOP:
            break

        scored = detector.score_row(event)
        results.append(scored)

        if scored["is_anomaly"]:
            print(
                "ANOMALY FLAG:",
                json.dumps({
                    "timestamp": str(scored["ts"]),
                    "machine_id": scored["machine_id"],
                    "score": round(scored["anomaly_score"], 4),
                }),
            )

    producer.join()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output, index=False)

    flagged = sum(row["is_anomaly"] for row in results)
    print("\nReplay finished.")
    print(f"Rows scored: {len(results):,}")
    print(f"Rows flagged: {flagged:,}")
    print(f"Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()