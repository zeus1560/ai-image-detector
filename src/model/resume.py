"""학습 재개 판단 (SPEC-02 처리 방식 5절).

torch를 import하지 않는 순수 함수만 둔다. 파일 읽기·쓰기와 torch 저장은 train.py가 한다.

저장 순서 (매 epoch 끝): metrics.csv 행 추가 → last.pt → (최소이면) best.pt
- last.pt가 "이 epoch까지 끝났다"는 기준이다.
- best는 val_loss 최소 epoch이고, 동점이면 앞 epoch다. train.py도 저장할 때
  best_epoch_of()로 같은 규칙을 써야 재개 시 정합성 검사와 어긋나지 않는다.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

KEEP_BEST = "keep"
REBUILD_BEST_FROM_LAST = "rebuild_from_last"


class ResumeError(Exception):
    """재개할 수 없는 불일치가 있을 때 나는 에러."""


@dataclass(frozen=True)
class ResumePlan:
    """재개 판단 결과.

    kept_rows: metrics.csv에 남길 (epoch, val_loss) 행 (입력 순서 유지)
    best_action: KEEP_BEST(그대로) 또는 REBUILD_BEST_FROM_LAST(last.pt 모델로 best.pt 다시 만들기)
    best_epoch: 정합성 확인 후 best.pt가 가리켜야 할 epoch
    """

    kept_rows: list
    best_action: str
    best_epoch: int


def best_epoch_of(rows) -> int:
    """(epoch, val_loss) 행 중 val_loss 최소 epoch. 동점이면 앞 epoch."""
    if not rows:
        raise ResumeError("metrics 행이 없어 best epoch를 정할 수 없습니다")
    for epoch, loss in rows:
        if not math.isfinite(loss):
            raise ResumeError(f"epoch {epoch}의 val_loss가 유한한 수가 아닙니다: {loss}")
    return min(rows, key=lambda r: (r[1], r[0]))[0]


def plan_resume(last_epoch: int, metrics_rows, best_epoch) -> ResumePlan:
    """재개 시 metrics 행 정리와 best.pt 정합성을 판단한다.

    Args:
        last_epoch: last.pt에 저장된 epoch (1부터)
        metrics_rows: metrics.csv의 (epoch, val_loss) 목록 (파일 순서)
        best_epoch: best.pt에 저장된 epoch. best.pt가 없으면 None

    처리
    1. last_epoch보다 뒤쪽 행을 지운다 (last.pt 저장 전에 끊긴 epoch의 행).
    2. 남은 행이 1~last_epoch를 정확히 한 번씩 갖지 않으면 에러.
    3. val_loss 최소 epoch(동점이면 앞)를 구한다.
       - best_epoch와 같으면 KEEP_BEST
       - 다르고, 최소 epoch == last_epoch이면 REBUILD_BEST_FROM_LAST (last.pt 후 best.pt 전에 끊긴 경우)
       - 그 외 불일치는 에러
    """
    if not isinstance(last_epoch, int) or isinstance(last_epoch, bool) or last_epoch < 1:
        raise ResumeError(f"last.pt의 epoch는 1 이상의 정수여야 합니다: {last_epoch!r}")

    rows = [(int(e), float(v)) for e, v in metrics_rows]
    kept = [r for r in rows if r[0] <= last_epoch]

    counts = Counter(e for e, _ in kept)
    expected = set(range(1, last_epoch + 1))
    missing = sorted(expected - counts.keys())
    duplicated = sorted(e for e, c in counts.items() if c > 1)
    unexpected = sorted(e for e in counts if e not in expected)
    if missing or duplicated or unexpected:
        raise ResumeError(
            f"metrics.csv가 1~{last_epoch} epoch를 한 번씩 갖지 않습니다 "
            f"(누락 {missing}, 중복 {duplicated}, 범위 밖 {unexpected})"
        )

    min_epoch = best_epoch_of(kept)
    if best_epoch == min_epoch:
        return ResumePlan(kept, KEEP_BEST, min_epoch)
    if min_epoch == last_epoch:
        return ResumePlan(kept, REBUILD_BEST_FROM_LAST, min_epoch)
    raise ResumeError(
        f"best.pt 불일치: best.pt epoch={best_epoch}, metrics의 val_loss 최소 epoch={min_epoch}, "
        f"last.pt epoch={last_epoch}"
    )
