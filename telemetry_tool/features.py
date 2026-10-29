import pandas as pd

from .data_loader import SENSOR_COLUMNS


def build_features(
    data: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Create sensor and short-term trend features."""
    if window < 2:
        raise ValueError("window must be at least 2")

    features = pd.DataFrame(index=data.index)

    for sensor in SENSOR_COLUMNS:
        values = pd.to_numeric(
            data[sensor], errors="coerce"
        )

        features[f"{sensor}_value"] = values

        features[f"{sensor}_rolling_mean"] = (
            values.rolling(window, min_periods=2).mean()
        )

        features[f"{sensor}_rolling_std"] = (
            values.rolling(window, min_periods=2).std()
        )

        features[f"{sensor}_change"] = values.diff()

    # Prevent infinite values from entering the ML model.
    features = features.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    # Fill missing feature values using the available data's
    # median. The original sensor columns remain unchanged.
    medians = features.median().fillna(0)

    return features.fillna(medians).fillna(0)