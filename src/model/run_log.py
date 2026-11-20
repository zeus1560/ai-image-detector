"""metrics.csv 열 정의와 읽기·쓰기 (SPEC-02 확정 후 보완 2-2).

torch를 import하지 않는다. train.py는 이 모듈로만 metrics.csv를 읽고 쓴다.

- 값이 None인 지표는 빈칸으로 저장하고, 읽을 때 빈칸은 None으로 돌린다.
- epoch와 val_loss는 빈칸이면 에러 (재개 판단과 best 선택에 꼭 필요하다).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

# 지표 열 이름은 metrics.py label_metrics의 키에 val_을 붙인 것
COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_false_positive_rate",
    "val_false_negative_rate",
    "val_ap",
    "val_auc",
    "epoch_seconds",
]

# 빈칸(None)을 허용하지 않는 열
NOT_NULL_COLUMNS = ("epoch", "val_loss")


class RunLogError(ValueError):
    """metrics.csv 형식이 열 정의와 맞지 않을 때 나는 에러."""


def _check_row_keys(row: dict) -> None:
    missing = [c for c in COLUMNS if c not in row]
    extra = [k for k in row if k not in COLUMNS]
    if missing or extra:
        raise RunLogError(f"metrics 행의 열이 맞지 않습니다 (빠짐 {missing}, 남음 {extra})")
    for c in NOT_NULL_COLUMNS:
        if row[c] is None:
            raise RunLogError(f"{c}는 비울 수 없습니다 (epoch {row['epoch']})")


def _format(value) -> str:
    return "" if value is None else str(value)


def _write_rows(f, rows, with_header: bool) -> None:
    writer = csv.writer(f, lineterminator="\n")
    if with_header:
        writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([_format(row[c]) for c in COLUMNS])


def append_row(path, row: dict) -> None:
    """metrics.csv에 한 행을 추가한다. 파일이 없으면 헤더와 함께 만든다."""
    _check_row_keys(row)
    path = Path(path)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        _write_rows(f, [row], with_header=is_new)


def _parse(column: str, text: str, line: int):
    if text == "":
        if column in NOT_NULL_COLUMNS:
            raise RunLogError(f"metrics.csv {line}번 줄: {column}가 비어 있습니다")
        return None
    try:
        return int(text) if column == "epoch" else float(text)
    except ValueError:
        raise RunLogError(f"metrics.csv {line}번 줄: {column} 값을 숫자로 읽을 수 없습니다: {text!r}") from None


def read_rows(path) -> list[dict]:
    """metrics.csv를 읽어 행(dict) 목록을 파일 순서대로 돌려준다. 헤더가 열 정의와 다르면 에러."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != COLUMNS:
            raise RunLogError(f"metrics.csv 헤더가 다릅니다: {header} (기대 {COLUMNS})")
        rows = []
        for line, values in enumerate(reader, start=2):
            if len(values) != len(COLUMNS):
                raise RunLogError(f"metrics.csv {line}번 줄: 값 {len(values)}개 (기대 {len(COLUMNS)}개)")
            rows.append({c: _parse(c, v, line) for c, v in zip(COLUMNS, values)})
    return rows


def loss_rows(rows) -> list[tuple[int, float]]:
    """plan_resume·is_new_best에 넘길 (epoch, val_loss) 목록."""
    return [(r["epoch"], r["val_loss"]) for r in rows]


def rewrite_kept(path, rows, kept_rows) -> None:
    """재개 시 plan_resume이 남긴 epoch의 행만 원래 순서대로 다시 쓴다 (임시 파일 후 os.replace).

    Args:
        rows: read_rows로 읽은 전체 행
        kept_rows: plan_resume 결과의 kept_rows ((epoch, val_loss) 목록)
    """
    kept_epochs = {e for e, _ in kept_rows}
    kept = [r for r in rows if r["epoch"] in kept_epochs]
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        _write_rows(f, kept, with_header=True)
    os.replace(tmp, path)
