from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data_loader import SENSOR_COLUMNS
from .features import build_features


class StreamingAnomalyDetector:
    """Score incoming telemetry rows against a historical baseline."""

    def __init__(
        self,
        contamination=0.01,
        window=5,
        random_state=42,
    ):
        if window < 2:
            raise ValueError("window must be at least 2")

        self.contamination = contamination
        self.window = window
        self.random_state = random_state
        self.models = {}
        self.medians = {}
        self.history = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self.feature_columns = None

    def fit(self, historical_data):
        """Train one model per machine using historical telemetry."""
        if historical_data.empty:
            raise ValueError("Historical data cannot be empty")

        self.models.clear()
        self.medians.clear()
        self.history.clear()

        for machine_id, machine in historical_data.groupby(
            "machine_id", sort=False
        ):
            machine = machine.sort_values("ts").copy()
            features = build_features(
                machine, window=self.window
            )

            medians = features.median().fillna(0)
            features = features.fillna(medians).fillna(0)

            model = IsolationForest(
                n_estimators=150,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(features)

            self.models[machine_id] = model
            self.medians[machine_id] = medians
            self.feature_columns = list(features.columns)

            # Seed rolling history with the last historical readings.
            for _, row in machine.tail(self.window - 1).iterrows():
                self.history[machine_id].append(
                    row[SENSOR_COLUMNS].to_dict()
                )

        return self

    def _make_feature_row(self, machine_id, row):
        """Build features using previous readings plus this new row."""
        previous = list(self.history[machine_id])
        current = {
            sensor: pd.to_numeric(
                pd.Series([row.get(sensor)]),
                errors="coerce",
            ).iloc[0]
            for sensor in SENSOR_COLUMNS
        }

        # The current row participates in the rolling window.
        samples = previous + [current]
        samples = samples[-self.window:]

        features = {}

        for sensor in SENSOR_COLUMNS:
            values = pd.Series(
                [item.get(sensor, np.nan) for item in samples],
                dtype="float64",
            ).replace([np.inf, -np.inf], np.nan)

            clean = values.dropna()

            features[f"{sensor}_value"] = current[sensor]
            features[f"{sensor}_rolling_mean"] = (
                clean.mean() if len(clean) else np.nan
            )
            features[f"{sensor}_rolling_std"] = (
                clean.std() if len(clean) >= 2 else np.nan
            )

            previous_values = [
                item.get(sensor, np.nan) for item in previous
            ]
            previous_valid = [
                value for value in previous_values
                if pd.notna(value)
            ]

            features[f"{sensor}_change"] = (
                current[sensor] - previous_valid[-1]
                if pd.notna(current[sensor]) and previous_valid
                else np.nan
            )

        feature_row = pd.DataFrame([features])
        feature_row = feature_row.reindex(
            columns=self.feature_columns
        )
        feature_row = feature_row.replace(
            [np.inf, -np.inf], np.nan
        )
        feature_row = feature_row.fillna(
            self.medians[machine_id]
        ).fillna(0)

        return feature_row

    def score_row(self, row):
        """Score one new row and return it with an anomaly flag."""
        machine_id = row["machine_id"]

        if machine_id not in self.models:
            raise ValueError(
                f"No trained model for machine {machine_id!r}"
            )

        feature_row = self._make_feature_row(machine_id, row)
        model = self.models[machine_id]

        result = dict(row)
        result["anomaly_score"] = float(
            -model.decision_function(feature_row)[0]
        )
        result["is_anomaly"] = bool(
            model.predict(feature_row)[0] == -1
        )

        # Store this reading for the next incoming row.
        self.history[machine_id].append(
            {sensor: row.get(sensor, np.nan)
             for sensor in SENSOR_COLUMNS}
        )

        return result