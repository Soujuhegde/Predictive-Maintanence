import pandas as pd

from telemetry_tool import detect_anomalies
from telemetry_tool.data_loader import SENSOR_COLUMNS
from telemetry_tool.streaming import StreamingAnomalyDetector


def make_test_data(n=120):
    data = pd.DataFrame({
        "ts": pd.date_range(
            "2026-03-01",
            periods=n,
            freq="15s",
        ),
        "machine_id": ["TEST-01"] * n,
    })

    for sensor in SENSOR_COLUMNS:
        data[sensor] = 10.0

    # Create a mostly stable operating pattern.
    data["spindle_speed_rpm"] = [
        1000 + (i % 7) for i in range(n)
    ]
    data["spindle_load_pct"] = [
        40 + (i % 5) for i in range(n)
    ]
    data["spindle_current_a"] = [
        20 + (i % 4) for i in range(n)
    ]
    data["motor_temp_c"] = [
        65 + (i % 3) for i in range(n)
    ]
    data["vibration_mm_s"] = [
        0.3 + (i % 4) * 0.01 for i in range(n)
    ]
    data["coolant_flow_lpm"] = [
        40 + (i % 3) for i in range(n)
    ]

    # Inject a deliberately unusual combination.
    data.loc[
        n - 1,
        [
            "spindle_load_pct",
            "spindle_current_a",
            "motor_temp_c",
            "vibration_mm_s",
        ],
    ] = [99, 90, 150, 8]

    return data


def test_detector_returns_scores_and_flags():
    result = detect_anomalies(
        make_test_data(),
        contamination=0.03,
        window=5,
    )

    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns
    assert result["anomaly_score"].notna().all()
    assert result["is_anomaly"].dtype == bool


def test_extreme_pattern_is_flagged():
    result = detect_anomalies(
        make_test_data(),
        contamination=0.03,
        window=5,
    )

    assert bool(result.iloc[-1]["is_anomaly"]) is True


def test_streaming_detector_scores_and_records_row():
    historical = make_test_data().iloc[:-1].copy()
    incoming = make_test_data(n=2).iloc[-1].to_dict()

    detector = StreamingAnomalyDetector(
        contamination=0.03,
        window=5,
    ).fit(historical)

    result = detector.score_row(incoming)

    assert isinstance(result["is_anomaly"], bool)
    assert isinstance(result["anomaly_score"], float)
    assert len(detector.history["TEST-01"]) == detector.window