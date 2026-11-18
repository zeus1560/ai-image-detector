"""SPEC-02 A: metrics.py 테스트 (손 계산 값과 비교)."""

import pytest

from src.model.metrics import average_precision, label_metrics, roc_auc


# ---- 라벨 기준 지표 (A7) ----

def test_label_metrics_hand_calculated():
    # real 3장 중 1장을 fake로 예측 → 오탐률 1/3
    # fake 2장 중 1장을 real로 예측 → 미탐률 1/2
    # 맞힌 수 3/5
    m = label_metrics([0, 0, 0, 1, 1], [0, 1, 0, 1, 0])
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["false_negative_rate"] == pytest.approx(1 / 2)


def test_label_metrics_no_real_gives_none_fpr():
    m = label_metrics([1, 1], [1, 0])
    assert m["false_positive_rate"] is None
    assert m["false_negative_rate"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(1 / 2)


def test_label_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        label_metrics([0, 2], [0, 1])
    with pytest.raises(ValueError):
        label_metrics([0, 1], [0])


# ---- 점수 기준 지표 (A8) ----
# 손 계산 예시 (같은 점수 0.8 포함)
#   점수  0.9  0.8  0.8  0.5  0.2
#   정답  1    1    0    0    1      (fake 3장, real 2장)
#
# AP: 임계값별 (TP, FP, 정밀도 P, 재현율 R)
#   ≥0.9: (1,0) P=1    R=1/3
#   ≥0.8: (2,1) P=2/3  R=2/3   ← 0.8 두 장은 한 임계값으로 묶음
#   ≥0.5: (2,2) P=1/2  R=2/3
#   ≥0.2: (3,2) P=3/5  R=1
#   AP = 1/3·1 + 1/3·2/3 + 0·1/2 + 1/3·3/5 = 15/45 + 10/45 + 9/45 = 34/45
#
# ROC-AUC: (fake, real) 쌍 6개
#   0.9 vs 0.8 → 1, 0.9 vs 0.5 → 1
#   0.8 vs 0.8 → 0.5, 0.8 vs 0.5 → 1
#   0.2 vs 0.8 → 0, 0.2 vs 0.5 → 0
#   AUC = 3.5 / 6 = 7/12
Y = [1, 1, 0, 0, 1]
S = [0.9, 0.8, 0.8, 0.5, 0.2]


def test_average_precision_hand_calculated_with_ties():
    assert average_precision(Y, S) == pytest.approx(34 / 45)


def test_roc_auc_hand_calculated_with_ties():
    assert roc_auc(Y, S) == pytest.approx(7 / 12)


def test_score_metrics_order_independent():
    """입력 순서를 바꿔도 같은 값 (정렬을 제대로 하는지)."""
    order = [4, 2, 0, 3, 1]
    y = [Y[i] for i in order]
    s = [S[i] for i in order]
    assert average_precision(y, s) == pytest.approx(34 / 45)
    assert roc_auc(y, s) == pytest.approx(7 / 12)


def test_score_metrics_extremes():
    # 완전히 분리: AP=1, AUC=1
    assert average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    # 완전히 반대: AUC=0
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)
    # 모두 같은 점수: AUC=0.5, AP=양성 비율(임계값 하나에서 P=2/4, R=1)
    assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)
    assert average_precision([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)


# ---- 한 라벨만 있을 때 (A9) ----

@pytest.mark.parametrize("y", [[0, 0, 0], [1, 1, 1]])
def test_score_metrics_single_label_none(y):
    assert average_precision(y, [0.1, 0.5, 0.9]) is None
    assert roc_auc(y, [0.1, 0.5, 0.9]) is None


def test_score_metrics_reject_nan():
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1, float("nan")])
