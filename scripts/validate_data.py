#!/usr/bin/env python3
"""Validate the synthetic traffic datasets used by the ETA notebook."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EXPECTED_COLUMNS = {
    "traffic_data.csv": {"road", "timestamp", "avg_speed"},
    "weather_data.csv": {"timestamp", "rain", "temp", "humidity"},
    "events_data.csv": {"timestamp", "event_type"},
}


class ValidationError(Exception):
    """Raised when one or more dataset checks fail."""


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise ValidationError(f"Missing required file: {path}")
    return pd.read_csv(path)


def validate_columns(name: str, df: pd.DataFrame) -> None:
    expected = EXPECTED_COLUMNS[name]
    actual = set(df.columns)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        bits = []
        if missing:
            bits.append(f"missing columns={sorted(missing)}")
        if extra:
            bits.append(f"unexpected columns={sorted(extra)}")
        raise ValidationError(f"{name}: schema mismatch ({', '.join(bits)})")


def validate_traffic(df: pd.DataFrame) -> None:
    if (df["avg_speed"] <= 0).any():
        raise ValidationError("traffic_data.csv: avg_speed must be strictly positive")
    if df["road"].isna().any():
        raise ValidationError("traffic_data.csv: road contains null values")


def validate_weather(df: pd.DataFrame) -> None:
    if (df["humidity"].lt(0) | df["humidity"].gt(100)).any():
        raise ValidationError("weather_data.csv: humidity must be within [0, 100]")
    if (df["temp"].lt(-20) | df["temp"].gt(60)).any():
        raise ValidationError("weather_data.csv: temp outside plausible range [-20, 60]")
    if (df["rain"] < 0).any():
        raise ValidationError("weather_data.csv: rain cannot be negative")


def validate_events(df: pd.DataFrame) -> None:
    if df["event_type"].isna().any():
        raise ValidationError("events_data.csv: event_type contains null values")


def validate_mergeability(traffic: pd.DataFrame, weather: pd.DataFrame, events: pd.DataFrame) -> None:
    traffic_ts = set(pd.to_datetime(traffic["timestamp"]))
    weather_ts = set(pd.to_datetime(weather["timestamp"]))
    events_ts = set(pd.to_datetime(events["timestamp"]))

    missing_weather = len(traffic_ts - weather_ts)
    if missing_weather:
        raise ValidationError(
            f"mergeability check failed: {missing_weather} traffic timestamps absent from weather data"
        )

    if not events_ts.issubset(traffic_ts):
        extra = len(events_ts - traffic_ts)
        raise ValidationError(
            f"mergeability check failed: {extra} event timestamps absent from traffic data"
        )


def main() -> int:
    try:
        traffic = load_csv("traffic_data.csv")
        weather = load_csv("weather_data.csv")
        events = load_csv("events_data.csv")

        validate_columns("traffic_data.csv", traffic)
        validate_columns("weather_data.csv", weather)
        validate_columns("events_data.csv", events)

        validate_traffic(traffic)
        validate_weather(weather)
        validate_events(events)
        validate_mergeability(traffic, weather, events)

        print("✅ Dataset validation passed.")
        print(
            f"Rows: traffic={len(traffic)}, weather={len(weather)}, events={len(events)}"
        )
        return 0
    except ValidationError as exc:
        print(f"❌ Validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
