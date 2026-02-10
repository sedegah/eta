#!/usr/bin/env python3
"""Validate synthetic traffic datasets used by the ETA notebook.

This validator is intentionally dependency-free (stdlib only) so it can run
before installing notebook/ML packages.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EXPECTED_COLUMNS = {
    "traffic_data.csv": ["road", "timestamp", "avg_speed"],
    "weather_data.csv": ["timestamp", "rain", "temp", "humidity"],
    "events_data.csv": ["timestamp", "event_type"],
}

KNOWN_ROADS = {"Circle Rd", "Spintex Rd", "Independence Ave"}
KNOWN_EVENT_TYPES = {"none", "concert", "sports", "festival", "market_day", "accident"}
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class ValidationError(Exception):
    """Raised when one or more dataset checks fail."""


def load_csv(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / name
    if not path.exists():
        raise ValidationError(f"Missing required file: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"{name}: missing header row")
        validate_columns(name, reader.fieldnames)
        return list(reader)


def validate_columns(name: str, headers: Iterable[str]) -> None:
    expected = EXPECTED_COLUMNS[name]
    actual = list(headers)

    if actual != expected:
        expected_set = set(expected)
        actual_set = set(actual)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValidationError(
            f"{name}: schema mismatch (expected order={expected}; got={actual}; "
            f"missing={missing}; unexpected={extra})"
        )


def parse_timestamp(raw: str, *, file_name: str, row_idx: int) -> datetime:
    if not raw:
        raise ValidationError(f"{file_name}: row {row_idx} has empty timestamp")
    try:
        return datetime.strptime(raw, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValidationError(
            f"{file_name}: row {row_idx} has invalid timestamp '{raw}' "
            f"(expected {TIMESTAMP_FORMAT})"
        ) from exc


def require_non_empty(value: str, *, file_name: str, row_idx: int, field: str) -> None:
    if value is None or value == "":
        raise ValidationError(f"{file_name}: row {row_idx} has empty {field}")


def parse_float(value: str, *, file_name: str, row_idx: int, field: str) -> float:
    require_non_empty(value, file_name=file_name, row_idx=row_idx, field=field)
    try:
        return float(value)
    except ValueError as exc:
        raise ValidationError(
            f"{file_name}: row {row_idx} has non-numeric {field}='{value}'"
        ) from exc


def parse_int(value: str, *, file_name: str, row_idx: int, field: str) -> int:
    require_non_empty(value, file_name=file_name, row_idx=row_idx, field=field)
    try:
        return int(value)
    except ValueError as exc:
        raise ValidationError(
            f"{file_name}: row {row_idx} has non-integer {field}='{value}'"
        ) from exc


def validate_traffic(rows: list[dict[str, str]]) -> set[datetime]:
    seen_timestamps: set[datetime] = set()

    for idx, row in enumerate(rows, start=2):
        road = row["road"]
        require_non_empty(road, file_name="traffic_data.csv", row_idx=idx, field="road")
        if road not in KNOWN_ROADS:
            raise ValidationError(
                f"traffic_data.csv: row {idx} has unknown road '{road}'"
            )

        ts = parse_timestamp(row["timestamp"], file_name="traffic_data.csv", row_idx=idx)
        seen_timestamps.add(ts)

        avg_speed = parse_float(
            row["avg_speed"], file_name="traffic_data.csv", row_idx=idx, field="avg_speed"
        )
        if avg_speed <= 0:
            raise ValidationError(
                f"traffic_data.csv: row {idx} has non-positive avg_speed={avg_speed}"
            )

    return seen_timestamps


def validate_weather(rows: list[dict[str, str]]) -> set[datetime]:
    seen_timestamps: set[datetime] = set()

    for idx, row in enumerate(rows, start=2):
        ts = parse_timestamp(row["timestamp"], file_name="weather_data.csv", row_idx=idx)
        seen_timestamps.add(ts)

        rain = parse_float(row["rain"], file_name="weather_data.csv", row_idx=idx, field="rain")
        if rain < 0:
            raise ValidationError(f"weather_data.csv: row {idx} has negative rain={rain}")

        temp = parse_float(row["temp"], file_name="weather_data.csv", row_idx=idx, field="temp")
        if temp < -20 or temp > 60:
            raise ValidationError(
                f"weather_data.csv: row {idx} has out-of-range temp={temp}"
            )

        humidity = parse_int(
            row["humidity"], file_name="weather_data.csv", row_idx=idx, field="humidity"
        )
        if humidity < 0 or humidity > 100:
            raise ValidationError(
                f"weather_data.csv: row {idx} has out-of-range humidity={humidity}"
            )

    return seen_timestamps


def validate_events(rows: list[dict[str, str]]) -> set[datetime]:
    seen_timestamps: set[datetime] = set()

    for idx, row in enumerate(rows, start=2):
        ts = parse_timestamp(row["timestamp"], file_name="events_data.csv", row_idx=idx)
        seen_timestamps.add(ts)

        event_type = row["event_type"]
        require_non_empty(
            event_type, file_name="events_data.csv", row_idx=idx, field="event_type"
        )
        if event_type not in KNOWN_EVENT_TYPES:
            raise ValidationError(
                f"events_data.csv: row {idx} has unknown event_type '{event_type}'"
            )

    return seen_timestamps


def validate_mergeability(
    traffic_timestamps: set[datetime],
    weather_timestamps: set[datetime],
    event_timestamps: set[datetime],
) -> None:
    missing_weather = len(traffic_timestamps - weather_timestamps)
    if missing_weather:
        raise ValidationError(
            "mergeability check failed: "
            f"{missing_weather} traffic timestamps absent from weather data"
        )

    extra_events = len(event_timestamps - traffic_timestamps)
    if extra_events:
        raise ValidationError(
            "mergeability check failed: "
            f"{extra_events} event timestamps absent from traffic data"
        )


def main() -> int:
    try:
        traffic_rows = load_csv("traffic_data.csv")
        weather_rows = load_csv("weather_data.csv")
        event_rows = load_csv("events_data.csv")

        if not traffic_rows or not weather_rows:
            raise ValidationError("traffic_data.csv and weather_data.csv must not be empty")

        traffic_ts = validate_traffic(traffic_rows)
        weather_ts = validate_weather(weather_rows)
        event_ts = validate_events(event_rows)
        validate_mergeability(traffic_ts, weather_ts, event_ts)

        print("✅ Dataset validation passed.")
        print(
            f"Rows: traffic={len(traffic_rows)}, weather={len(weather_rows)}, events={len(event_rows)}"
        )
        return 0
    except ValidationError as exc:
        print(f"❌ Validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
