from math import isinf

from src.risk_metrics import (
    calculate_relative_speed,
    calculate_risk_level,
    calculate_thw,
    calculate_ttc,
)


def test_calculate_ttc_closing_object():
    assert calculate_ttc(20.0, -5.0) == 4.0


def test_calculate_ttc_non_closing_object():
    assert isinf(calculate_ttc(20.0, 2.0))


def test_calculate_thw():
    assert round(calculate_thw(20.0, 10.0), 2) == 2.0


def test_classify_risk_critical():
    assert calculate_risk_level(ttc=1.0, thw=0.6, distance_m=4.0, object_type="car") == "CRITICAL"


def test_pedestrian_higher_risk():
    vehicle_level = calculate_risk_level(ttc=4.0, thw=2.5, distance_m=18.0, object_type="car")
    pedestrian_level = calculate_risk_level(ttc=4.0, thw=2.5, distance_m=18.0, object_type="person")

    assert vehicle_level == "MEDIUM"
    assert pedestrian_level == "HIGH"


def test_relative_speed_positive_when_gap_increases():
    assert calculate_relative_speed(10.0, 14.0, 2.0) == 2.0
