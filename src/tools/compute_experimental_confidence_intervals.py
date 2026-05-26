#!/usr/bin/env python3
"""Compute room-level confidence intervals for the experimental summary table."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_DIR = Path("docs/full_circle_vlm_stage_averages")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR

ROOM_TYPES = (
    ("bedroom", "Спальня"),
    ("kitchen", "Кухня"),
    ("living_room", "Гостиная"),
)
ROOM_TYPE_LABELS = dict(ROOM_TYPES)
COMPLETE_STAGES = {"infinigen", "postprocessing", "suppliers"}

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    80: 1.990,
    100: 1.984,
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    source: str
    stage: str
    value_column: str


VLM_METRICS = (
    MetricSpec(
        key="vlm_before",
        label="VLM-оценка до исправлений",
        source="vlm_stage_room_averages.csv",
        stage="infinigen",
        value_column="total_score_mean",
    ),
    MetricSpec(
        key="vlm_after",
        label="VLM-оценка после исправлений",
        source="vlm_stage_room_averages.csv",
        stage="suppliers",
        value_column="total_score_mean",
    ),
    MetricSpec(
        key="layout",
        label="Расстановка",
        source="vlm_stage_room_averages.csv",
        stage="suppliers",
        value_column="layout_score_mean",
    ),
    MetricSpec(
        key="physical_correctness",
        label="Физ. корректность",
        source="vlm_stage_room_averages.csv",
        stage="suppliers",
        value_column="collision_score_mean",
    ),
    MetricSpec(
        key="asset_quality",
        label="Качество объектов",
        source="vlm_stage_room_averages.csv",
        stage="suppliers",
        value_column="asset_quality_score_mean",
    ),
)

TIME_METRIC = MetricSpec(
    key="generation_time_min",
    label="Ср. время, мин",
    source="generation_time_by_room.csv",
    stage="",
    value_column="generation_time_min",
)
ALL_METRICS = VLM_METRICS + (TIME_METRIC,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 95% t confidence intervals from room-level scores."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def t_critical_975(df: int) -> float:
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    larger = [k for k in T_CRITICAL_975 if k > df]
    if larger:
        return T_CRITICAL_975[min(larger)]
    return 1.96


def stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.fmean(values)
    if n <= 1:
        return {
            "n": float(n),
            "mean": mean,
            "sd": 0.0,
            "se": 0.0,
            "t_critical": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "ci95_half_width": 0.0,
        }
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    tcrit = t_critical_975(n - 1)
    half_width = tcrit * se
    return {
        "n": float(n),
        "mean": mean,
        "sd": sd,
        "se": se,
        "t_critical": tcrit,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci95_half_width": half_width,
    }


def complete_vlm_cases(vlm_rows: list[dict[str, str]]) -> set[str]:
    stages_by_case: dict[str, set[str]] = defaultdict(set)
    for row in vlm_rows:
        if row.get("room_type") not in ROOM_TYPE_LABELS:
            continue
        if not truthy(row.get("complete_9_frames")):
            continue
        stages_by_case[row["case"]].add(row["stage"])
    return {
        case
        for case, stages in stages_by_case.items()
        if COMPLETE_STAGES.issubset(stages)
    }


def collect_vlm_values(
    vlm_rows: list[dict[str, str]],
    complete_cases: set[str],
) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    metric_by_stage_col = {
        (metric.stage, metric.value_column): metric
        for metric in VLM_METRICS
    }
    for row in vlm_rows:
        room_type = row.get("room_type")
        case = row.get("case")
        if room_type not in ROOM_TYPE_LABELS or case not in complete_cases:
            continue
        if not truthy(row.get("complete_9_frames")):
            continue
        for (stage, column), metric in metric_by_stage_col.items():
            if row.get("stage") != stage:
                continue
            raw_value = row.get(column)
            if raw_value not in (None, ""):
                values[(room_type, metric.key)].append(float(raw_value))
    return values


def collect_time_values(
    time_rows: list[dict[str, str]],
    complete_cases: set[str],
) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in time_rows:
        room_type = row.get("room_type")
        case = row.get("case")
        if room_type not in ROOM_TYPE_LABELS or case not in complete_cases:
            continue
        raw_value = row.get(TIME_METRIC.value_column)
        if raw_value not in (None, ""):
            values[(room_type, TIME_METRIC.key)].append(float(raw_value))
    return values


def build_rows(
    value_map: dict[tuple[str, str], list[float]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for room_type, label in ROOM_TYPES + (("all", "Среднее"),):
        for metric in ALL_METRICS:
            if room_type == "all":
                values = []
                for source_room_type, _ in ROOM_TYPES:
                    values.extend(value_map[(source_room_type, metric.key)])
            else:
                values = value_map[(room_type, metric.key)]
            result = stats(values)
            rows.append(
                {
                    "room_type": room_type,
                    "room_type_ru": label,
                    "metric": metric.key,
                    "metric_ru": metric.label,
                    "source": metric.source,
                    "stage": metric.stage,
                    "value_column": metric.value_column,
                    "n": str(int(result["n"])),
                    "mean": f"{result['mean']:.6f}",
                    "sd": f"{result['sd']:.6f}",
                    "se": f"{result['se']:.6f}",
                    "t_critical": f"{result['t_critical']:.6f}",
                    "ci95_low": f"{result['ci95_low']:.6f}",
                    "ci95_high": f"{result['ci95_high']:.6f}",
                    "ci95_half_width": f"{result['ci95_half_width']:.6f}",
                }
            )
    return rows


def formatted_interval(row: dict[str, str]) -> str:
    return (
        f"{float(row['mean']):.2f} "
        f"[{float(row['ci95_low']):.2f}; {float(row['ci95_high']):.2f}]"
    )


def build_wide_rows(long_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows_by_room_metric = {
        (row["room_type"], row["metric"]): row
        for row in long_rows
    }
    wide_rows = []
    for room_type, label in ROOM_TYPES + (("all", "Среднее"),):
        first_metric = rows_by_room_metric[(room_type, ALL_METRICS[0].key)]
        wide_row = {
            "Тип комнаты": label,
            "Кол-во комнат": first_metric["n"],
        }
        for metric in ALL_METRICS:
            wide_row[metric.label] = formatted_interval(
                rows_by_room_metric[(room_type, metric.key)]
            )
        wide_rows.append(wide_row)
    return wide_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, wide_rows: list[dict[str, str]]) -> None:
    columns = ["Тип комнаты", "Кол-во комнат"] + [metric.label for metric in ALL_METRICS]
    lines = [
        "# Доверительные интервалы для итоговой таблицы",
        "",
        "Метод: 95% t-интервал по независимым комнатам. Для VLM одна точка - средняя оценка комнаты по 9 ракурсам; используются только complete-triplet кейсы bedroom/kitchen/living, чтобы совпасть с n=59 на слайде.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in wide_rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    lines.extend(
        [
            "",
            "Примечание: средние здесь пересчитаны из raw CSV в `docs/full_circle_vlm_stage_averages`. Если в презентации остаются старые средние со скриншота, интервалы нужно синхронизировать с тем же raw набором оценок, из которого были получены эти средние.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vlm_rows = read_csv(input_dir / "vlm_stage_room_averages.csv")
    time_rows = read_csv(input_dir / "generation_time_by_room.csv")
    complete_cases = complete_vlm_cases(vlm_rows)

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    values.update(collect_vlm_values(vlm_rows, complete_cases))
    for key, metric_values in collect_time_values(time_rows, complete_cases).items():
        values[key].extend(metric_values)

    long_rows = build_rows(values)
    wide_rows = build_wide_rows(long_rows)

    long_fields = [
        "room_type",
        "room_type_ru",
        "metric",
        "metric_ru",
        "source",
        "stage",
        "value_column",
        "n",
        "mean",
        "sd",
        "se",
        "t_critical",
        "ci95_low",
        "ci95_high",
        "ci95_half_width",
    ]
    wide_fields = ["Тип комнаты", "Кол-во комнат"] + [metric.label for metric in ALL_METRICS]

    write_csv(
        output_dir / "experimental_evaluation_confidence_intervals.csv",
        long_rows,
        long_fields,
    )
    write_csv(
        output_dir / "experimental_evaluation_confidence_intervals_slide_table.csv",
        wide_rows,
        wide_fields,
    )
    write_markdown(
        output_dir / "experimental_evaluation_confidence_intervals.md",
        wide_rows,
    )

    print(f"complete cases: {len(complete_cases)}")
    print(output_dir / "experimental_evaluation_confidence_intervals.csv")
    print(output_dir / "experimental_evaluation_confidence_intervals_slide_table.csv")
    print(output_dir / "experimental_evaluation_confidence_intervals.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
