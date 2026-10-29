import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import build_features


def detect_anomalies(
    data: pd.DataFrame,
    contamination: float = 0.01,
    window: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect unusual telemetry patterns independently per machine.

    Returns:
        Original rows plus:
        - anomaly_score: higher means more unusual
        - is_anomaly: True for model-flagged rows
    """
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    if data.empty:
        result = data.copy()
        result["anomaly_score"] = pd.Series(dtype=float)
        result["is_anomaly"] = pd.Series(dtype=bool)
        return result

    results = []

    for machine_id, machine_data in data.groupby(
        "machine_id", sort=False
    ):
        machine_data = machine_data.copy()

        features = build_features(
            machine_data,
            window=window,
        )

        model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(features)

        # sklearn's decision_function is lower for more unusual
        # observations. Negating it makes higher scores unusual.
        machine_data["anomaly_score"] = (
            -model.decision_function(features)
        )

        machine_data["is_anomaly"] = (
            model.predict(features) == -1
        )

        results.append(machine_data)

    return (
        pd.concat(results)
        .sort_index()
    )