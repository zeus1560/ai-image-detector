"""판별 지표 (SPEC-02 처리 방식 7절, SPEC-03에서 재사용).

numpy만 사용한다 (torch, sklearn 없음).
라벨: real=0, fake=1. 양성은 fake.
"""

from __future__ import annotations

import numpy as np


def _as_labels(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name}는 1차원 배열이어야 합니다 (shape={arr.shape})")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name}에는 0(real)과 1(fake)만 있어야 합니다")
    return arr.astype(np.int64)


def _ratio(numerator: int, denominator: int):
    """분모가 0이면 None."""
    return None if denominator == 0 else numerator / denominator


def label_metrics(y_true, y_pred) -> dict:
    """예측 라벨 기준 지표.

    반환 키
    - accuracy: 맞힌 수 / 전체
    - false_positive_rate: 실제 사진 오탐률 = 정답 real 중 fake로 예측한 비율
    - false_negative_rate: 생성 이미지 미탐률 = 정답 fake 중 real로 예측한 비율
    분모가 0이면 그 값은 None.
    """
    y_true = _as_labels(y_true, "y_true")
    y_pred = _as_labels(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true와 y_pred의 길이가 다릅니다: {len(y_true)} != {len(y_pred)}")

    real = y_true == 0
    fake = y_true == 1
    return {
        "accuracy": _ratio(int((y_true == y_pred).sum()), len(y_true)),
        "false_positive_rate": _ratio(int((y_pred[real] == 1).sum()), int(real.sum())),
        "false_negative_rate": _ratio(int((y_pred[fake] == 0).sum()), int(fake.sum())),
    }


def _check_scores(y_true, scores):
    y_true = _as_labels(y_true, "y_true")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != y_true.shape:
        raise ValueError(f"y_true와 scores의 길이가 다릅니다: {y_true.shape} != {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores에 NaN 또는 무한대가 있습니다")
    return y_true, scores


def _has_both_labels(y_true: np.ndarray) -> bool:
    return bool((y_true == 0).any() and (y_true == 1).any())


def average_precision(y_true, scores):
    """AP = Σ (Rₙ − Rₙ₋₁) × Pₙ.

    점수 내림차순으로 서로 다른 점수값마다 임계값(점수 ≥ 임계값이면 fake)을 둔다.
    같은 점수는 하나의 임계값으로 묶는다. 한 라벨만 있으면 None.
    """
    y_true, scores = _check_scores(y_true, scores)
    if not _has_both_labels(y_true):
        return None

    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # 같은 점수 묶음의 마지막 위치에서만 정밀도·재현율을 잰다
    last_of_group = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = tp[last_of_group]
    fp = fp[last_of_group]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y_true, scores):
    """ROC-AUC = 모든 (fake, real) 쌍에서 fake 점수가 크면 1, 같으면 0.5, 작으면 0의 평균.

    쌍을 직접 세지 않고 같은 점수에 평균 순위를 주는 순위 합(Mann–Whitney U)으로 계산한다
    (val 2만 장이면 쌍이 1억 개라 직접 세기 어렵다). 한 라벨만 있으면 None.
    """
    y_true, scores = _check_scores(y_true, scores)
    if not _has_both_labels(y_true):
        return None

    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)  # 각 점수 묶음의 마지막 순위 (1부터)
    starts = ends - counts + 1
    ranks = ((starts + ends) / 2.0)[inverse]

    n_fake = int((y_true == 1).sum())
    n_real = int((y_true == 0).sum())
    u = ranks[y_true == 1].sum() - n_fake * (n_fake + 1) / 2.0
    return float(u / (n_fake * n_real))
