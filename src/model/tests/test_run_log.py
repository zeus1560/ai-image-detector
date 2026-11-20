"""SPEC-02 확정 후 보완 2-2: run_log.py 테스트 (torch 없이)."""

import pytest

from src.model.resume import plan_resume
from src.model.run_log import (
    COLUMNS,
    RunLogError,
    append_row,
    loss_rows,
    read_rows,
    rewrite_kept,
)


def make_row(epoch, val_loss, **overrides):
    row = {
        "epoch": epoch,
        "train_loss": 0.6,
        "val_loss": val_loss,
        "val_accuracy": 0.75,
        "val_false_positive_rate": 0.2,
        "val_false_negative_rate": 0.3,
        "val_ap": 0.8,
        "val_auc": 0.85,
        "epoch_seconds": 12.5,
    }
    row.update(overrides)
    return row


def test_columns_order():
    assert COLUMNS == [
        "epoch", "train_loss", "val_loss", "val_accuracy", "val_false_positive_rate",
        "val_false_negative_rate", "val_ap", "val_auc", "epoch_seconds",
    ]


def test_append_creates_header_and_roundtrips(tmp_path):
    path = tmp_path / "metrics.csv"
    append_row(path, make_row(1, 0.5))
    append_row(path, make_row(2, 0.4))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert len(lines) == 3
    rows = read_rows(path)
    assert rows == [make_row(1, 0.5), make_row(2, 0.4)]
    assert isinstance(rows[0]["epoch"], int)
    assert loss_rows(rows) == [(1, 0.5), (2, 0.4)]


def test_none_metrics_roundtrip_as_none(tmp_path):
    """None 지표는 빈칸으로 저장되고, 읽은 뒤에도 None이다."""
    path = tmp_path / "metrics.csv"
    row = make_row(1, 0.5, val_false_positive_rate=None, val_ap=None, val_auc=None)
    append_row(path, row)
    assert path.read_text(encoding="utf-8").splitlines()[1] == "1,0.6,0.5,0.75,,0.3,,,12.5"
    assert read_rows(path) == [row]


def test_float_precision_preserved(tmp_path):
    path = tmp_path / "metrics.csv"
    append_row(path, make_row(1, 0.1 + 0.2))
    assert read_rows(path)[0]["val_loss"] == 0.1 + 0.2


def test_header_mismatch_raises(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,val_loss,train_loss\n1,0.5,0.6\n", encoding="utf-8")
    with pytest.raises(RunLogError, match="헤더"):
        read_rows(path)


@pytest.mark.parametrize("column", ["epoch", "val_loss"])
def test_blank_epoch_or_val_loss_raises_on_read(tmp_path, column):
    path = tmp_path / "metrics.csv"
    append_row(path, make_row(1, 0.5))
    lines = path.read_text(encoding="utf-8").splitlines()
    values = lines[1].split(",")
    values[COLUMNS.index(column)] = ""
    path.write_text(lines[0] + "\n" + ",".join(values) + "\n", encoding="utf-8")
    with pytest.raises(RunLogError, match=column):
        read_rows(path)


@pytest.mark.parametrize("column", ["epoch", "val_loss"])
def test_none_epoch_or_val_loss_raises_on_append(tmp_path, column):
    with pytest.raises(RunLogError, match=column):
        row = make_row(1, 0.5)
        row[column] = None
        append_row(tmp_path / "metrics.csv", row)


def test_missing_or_extra_column_raises(tmp_path):
    path = tmp_path / "metrics.csv"
    row = make_row(1, 0.5)
    del row["val_auc"]
    with pytest.raises(RunLogError, match="val_auc"):
        append_row(path, row)
    with pytest.raises(RunLogError, match="extra"):
        append_row(path, make_row(1, 0.5, extra=1))
    assert not path.exists()


def test_rewrite_after_resume_keeps_1_to_last(tmp_path):
    """재개 다시 쓰기: last.pt 저장 전에 끊긴 epoch 3 행을 지우고 1~last(2) 행만 원래 순서대로 남긴다."""
    path = tmp_path / "metrics.csv"
    originals = [make_row(1, 0.5), make_row(2, 0.4), make_row(3, 0.3)]
    for r in originals:
        append_row(path, r)
    rows = read_rows(path)
    plan = plan_resume(2, loss_rows(rows), best_epoch=2)
    rewrite_kept(path, rows, plan.kept_rows)
    after = read_rows(path)
    assert [r["epoch"] for r in after] == [1, 2]
    assert after == originals[:2]
    assert not (tmp_path / "metrics.csv.tmp").exists()
    # 다시 쓴 뒤에도 이어서 추가할 수 있다
    append_row(path, make_row(3, 0.35))
    assert [r["epoch"] for r in read_rows(path)] == [1, 2, 3]
