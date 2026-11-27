"""SPEC-03 A: eval_report.py 테스트.

기대값은 모두 손으로 계산한 숫자다 (코드로 다시 계산하지 않는다).
"""

import math

import pandas as pd
import pytest

from src.model.eval_report import (
    BASELINE_ALL,
    BASELINE_LINKED,
    BASELINE_NONE,
    BASELINE_SAME_SOURCE,
    PRED_COLUMNS,
    KakaoLinkError,
    build_predictions,
    compute_drop,
    format_bias_warnings,
    read_predictions,
    summarize,
    wilson_interval,
    write_predictions,
)

# ---------------------------------------------------------------- 손으로 만든 예측 표


def make_predictions(groups):
    """groups: [(source, condition, [(origin_id, label, generator, pred[, gray]), ...]), ...]

    로짓은 pred가 1이면 양수, 0이면 음수. 행마다 크기를 조금씩 다르게 준다.
    gray는 생략하면 False (SPEC-03 보완 2에서 predictions에 gray 열 추가).
    """
    rows = []
    for source, condition, items in groups:
        for oid, label, generator, pred, *rest in items:
            gray = rest[0] if rest else False
            i = len(rows)
            logit = (1.0 + 0.1 * i) * (1 if pred == 1 else -1)
            rows.append(
                {
                    "row_index": i,
                    "path": f"images/{source}/{condition}/{oid}.png",
                    "label": label,
                    "source": source,
                    "generator": generator,
                    "condition": condition,
                    "origin_id": oid,
                    "logit": logit,
                    "score": 1 / (1 + math.exp(-logit)),
                    "pred": pred,
                    "gray": gray,
                }
            )
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def items(prefix, real_preds, fake_preds_by_gen):
    """real 행과 생성기별 fake 행. origin_id는 prefix+번호."""
    out = [(f"{prefix}r{i}", 0, "real", p) for i, p in enumerate(real_preds)]
    for gen, preds in fake_preds_by_gen.items():
        out += [(f"{prefix}{gen}{i}", 1, gen, p) for i, p in enumerate(preds)]
    return out


# source 2개, condition 3개, generator 2개 (sd14, mj)
HAND_TABLE = make_predictions(
    [
        ("genimage", "original", items("o", [0, 0, 0, 1], {"sd14": [1, 1, 1], "mj": [1, 1, 0]})),
        ("genimage", "jpeg75", items("o", [1, 1, 0, 0], {"sd14": [1, 1, 0], "mj": [1, 0, 0]})),
        ("genimage", "jpeg50", items("o", [0, 0, 0, 0], {"sd14": [0, 0, 0], "mj": [0, 0, 0]})),
        ("cifake", "original", items("c", [0, 0], {"sd14": [1, 1]})),
        ("cifake", "jpeg75", items("c", [1, 0], {"sd14": [1, 0]})),
        ("cifake", "jpeg50", items("c", [1, 1], {"sd14": [0, 0]})),
    ]
)

# (source, condition): n_real, n_fake, 오탐 수, 미탐 수, 정확도, 오탐률, 미탐률 (손 계산)
EXPECTED_CONDITION = {
    ("genimage", "original"): (4, 6, 1, 1, 8 / 10, 1 / 4, 1 / 6),
    ("genimage", "jpeg75"): (4, 6, 2, 3, 5 / 10, 2 / 4, 3 / 6),
    ("genimage", "jpeg50"): (4, 6, 0, 6, 4 / 10, 0 / 4, 6 / 6),
    ("cifake", "original"): (2, 2, 0, 0, 4 / 4, 0 / 2, 0 / 2),
    ("cifake", "jpeg75"): (2, 2, 1, 1, 2 / 4, 1 / 2, 1 / 2),
    ("cifake", "jpeg50"): (2, 2, 2, 2, 0 / 4, 2 / 2, 2 / 2),
}

# (source, condition, generator): n_fake, 미탐 수, 미탐률 (손 계산)
EXPECTED_GENERATOR = {
    ("genimage", "original", "sd14"): (3, 0, 0 / 3),
    ("genimage", "original", "mj"): (3, 1, 1 / 3),
    ("genimage", "jpeg75", "sd14"): (3, 1, 1 / 3),
    ("genimage", "jpeg75", "mj"): (3, 2, 2 / 3),
    ("genimage", "jpeg50", "sd14"): (3, 3, 3 / 3),
    ("genimage", "jpeg50", "mj"): (3, 3, 3 / 3),
    ("cifake", "original", "sd14"): (2, 0, 0 / 2),
    ("cifake", "jpeg75", "sd14"): (2, 1, 1 / 2),
    ("cifake", "jpeg50", "sd14"): (2, 2, 2 / 2),
}

# 보완 2: 생성기 행 = 그 (source, condition)의 공유 실제 전체 + 그 생성기 가짜 (손 표는 모두 gray=False)
# (source, condition, generator): n_real, 오탐 수, 정확도, 오탐률, AP, ROC-AUC (손 계산)
# 점수 순서는 make_predictions의 로짓 (1+0.1i)×(±1)로 정해진다
EXPECTED_GENERATOR_FULL = {
    # 실제 -1.0 -1.1 -1.2 +1.3 / sd14 +1.4 +1.5 +1.6 → 가짜가 모든 실제보다 큼
    ("genimage", "original", "sd14"): (4, 1, 6 / 7, 1 / 4, 1.0, 1.0),
    # mj +1.7 +1.8 −1.9: AUC 8/12, AP = 1/3 + 1/3 + (1/3)(3/7)
    ("genimage", "original", "mj"): (4, 1, 5 / 7, 1 / 4, 17 / 21, 2 / 3),
    # 실제 +2.0 +2.1 −2.2 −2.3 / sd14 +2.4 +2.5 −2.6
    ("genimage", "jpeg75", "sd14"): (4, 2, 4 / 7, 2 / 4, 17 / 21, 2 / 3),
    # mj +2.7 −2.8 −2.9: AUC 4/12, AP = 1/3 + (1/3)(2/6) + (1/3)(3/7)
    ("genimage", "jpeg75", "mj"): (4, 2, 3 / 7, 2 / 4, 37 / 63, 1 / 3),
    # 실제 −3.0~−3.3이 모든 가짜보다 큼: AUC 0, AP = (1/3)(1/5 + 2/6 + 3/7)
    ("genimage", "jpeg50", "sd14"): (4, 0, 4 / 7, 0 / 4, 101 / 315, 0.0),
    ("genimage", "jpeg50", "mj"): (4, 0, 4 / 7, 0 / 4, 101 / 315, 0.0),
    ("cifake", "original", "sd14"): (2, 0, 4 / 4, 0 / 2, 1.0, 1.0),
    # 실제 +4.4 −4.5 / sd14 +4.6 −4.7: AUC 2/4, AP = 1/2 + (1/2)(2/4)
    ("cifake", "jpeg75", "sd14"): (2, 1, 2 / 4, 1 / 2, 3 / 4, 1 / 2),
    # 실제 +4.8 +4.9 / sd14 −5.0 −5.1: AUC 0, AP = (1/2)(1/3) + (1/2)(2/4)
    ("cifake", "jpeg50", "sd14"): (2, 2, 0 / 4, 2 / 2, 5 / 12, 0.0),
}


# ---------------------------------------------------------------- summary


def test_summary_condition_rows_match_hand_calc():
    """A2: (source, condition) 행마다 개수와 정확도·오탐률·미탐률이 손 계산과 같다."""
    summary = summarize(HAND_TABLE)
    cond = summary[summary["group"] == "condition"]
    assert len(cond) == len(EXPECTED_CONDITION)
    for _, r in cond.iterrows():
        n_real, n_fake, n_fp, n_fn, acc, fpr, fnr = EXPECTED_CONDITION[(r["source"], r["condition"])]
        assert (r["n_real"], r["n_fake"], r["n_false_positive"], r["n_false_negative"]) == (n_real, n_fake, n_fp, n_fn)
        assert r["accuracy"] == pytest.approx(acc)
        assert r["false_positive_rate"] == pytest.approx(fpr)
        assert r["false_negative_rate"] == pytest.approx(fnr)
        assert r["generator"] is None
        # 두 라벨이 다 있으므로 AP·AUC와 Wilson 구간이 채워진다
        assert r["ap"] is not None and r["auc"] is not None
        assert r["accuracy_ci_low"] <= r["accuracy"] <= r["accuracy_ci_high"]

    # original이 각 source의 맨 앞
    assert list(cond["condition"])[:1] == ["original"]


def test_summary_generator_rows_shared_real_full_metrics():
    """A3: 생성기별 행은 generator≠real만 있고, 미탐률이 손 계산과 같다.

    보완 2: 생성기 행은 공유 실제 + 그 생성기 가짜로 전체 지표를 채운다 (손 계산 값과 같다).
    공유 실제를 같이 쓰므로 같은 (source, condition)의 생성기 행끼리 오탐률이 같다.
    """
    summary = summarize(HAND_TABLE)
    gen = summary[summary["group"] == "generator"]
    assert "real" not in set(gen["generator"])
    assert len(gen) == len(EXPECTED_GENERATOR)
    for _, r in gen.iterrows():
        key = (r["source"], r["condition"], r["generator"])
        n_fake, n_fn, fnr = EXPECTED_GENERATOR[key]
        assert (r["n_fake"], r["n_false_negative"]) == (n_fake, n_fn)
        assert r["false_negative_rate"] == pytest.approx(fnr)
        # 생성기별 행에도 오탐률·정확도·AP·AUC가 채워져 있고 손 계산 값과 같다
        n_real, n_fp, acc, fpr, ap, auc = EXPECTED_GENERATOR_FULL[key]
        assert (r["n_real"], r["n_false_positive"]) == (n_real, n_fp)
        assert r["false_positive_rate"] == pytest.approx(fpr)
        assert r["accuracy"] == pytest.approx(acc)
        assert r["ap"] == pytest.approx(ap) and r["auc"] == pytest.approx(auc)
    # 같은 (source, condition)의 생성기 행끼리 오탐률이 같다
    for _, g in gen.groupby(["source", "condition"]):
        assert len(set(g["false_positive_rate"])) == 1


def test_single_label_group_blanks():
    """A4: fake만 있는 그룹은 오탐률·AP·ROC-AUC(와 오탐률 구간)가 빈칸이다."""
    table = make_predictions([("latest", "original", items("l", [], {"flux": [1, 1, 0]}))])
    r = summarize(table).iloc[0]
    assert r["n_real"] == 0 and r["n_fake"] == 3
    assert r["false_positive_rate"] is None
    assert r["false_positive_rate_ci_low"] is None and r["false_positive_rate_ci_high"] is None
    assert r["ap"] is None and r["auc"] is None
    assert r["accuracy"] == pytest.approx(2 / 3)
    assert r["false_negative_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- Wilson


def test_wilson_matches_hand_calc():
    """A10: 100장 중 90 맞음의 Wilson 95% 구간 = (0.8256, 0.9448) (손 계산, 소수 4자리)."""
    low, high = wilson_interval(90, 100)
    assert round(low, 4) == 0.8256
    assert round(high, 4) == 0.9448


def test_wilson_empty_and_edges():
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0 < high < 1
    low, high = wilson_interval(10, 10)
    assert 0 < low < 1 and high == 1.0


# ---------------------------------------------------------------- drop


def drop_row(drop, source, condition):
    # 보완 2: drop.csv에 generator 행이 생겨 같은 (source, condition)에 여러 행이 있으므로 condition 행만 본다
    r = drop[(drop["group"] == "condition") & (drop["source"] == source) & (drop["condition"] == condition)]
    assert len(r) == 1
    return r.iloc[0]


def test_drop_matches_hand_calc_in_pp():
    """A5: 정확도·오탐률·미탐률 하락이 %p 손 계산 값과 같다."""
    drop = compute_drop(HAND_TABLE)
    assert (drop["group"] == "condition").sum() == 4  # original 행은 없다
    # genimage original: 정확도 0.8, 오탐률 0.25, 미탐률 1/6
    r = drop_row(drop, "genimage", "jpeg75")
    assert r["baseline"] == BASELINE_SAME_SOURCE
    assert r["accuracy_drop_pp"] == pytest.approx(30.0)  # 80 − 50
    assert r["false_positive_rate_increase_pp"] == pytest.approx(25.0)  # 50 − 25
    assert r["false_negative_rate_increase_pp"] == pytest.approx(100 / 3)  # 50 − 16.67
    r = drop_row(drop, "genimage", "jpeg50")
    assert r["accuracy_drop_pp"] == pytest.approx(40.0)  # 80 − 40
    assert r["false_positive_rate_increase_pp"] == pytest.approx(-25.0)  # 0 − 25
    assert r["false_negative_rate_increase_pp"] == pytest.approx(250 / 3)  # 100 − 16.67
    r = drop_row(drop, "cifake", "jpeg50")
    assert r["accuracy_drop_pp"] == pytest.approx(100.0)
    assert r["false_positive_rate_increase_pp"] == pytest.approx(100.0)
    assert r["false_negative_rate_increase_pp"] == pytest.approx(100.0)


def test_drop_blank_when_source_has_no_original():
    """A5: original이 없는 source는 drop 값이 빈칸이다."""
    table = make_predictions([("latest", "jpeg50", items("l", [0, 1], {"flux": [1, 0]}))])
    r = drop_row(compute_drop(table), "latest", "jpeg50")
    assert r["baseline"] == BASELINE_NONE
    assert r["accuracy_drop_pp"] is None
    assert r["false_positive_rate_increase_pp"] is None
    assert r["false_negative_rate_increase_pp"] is None
    assert r["ap_drop_pp"] is None and r["auc_drop_pp"] is None


# kakao 기준이 될 genimage original: 정확도 6/8, 오탐률 1/4, 미탐률 1/4
GENIMAGE_ORIGINAL = (
    "genimage",
    "original",
    [
        ("r1", 0, "real", 0),
        ("r2", 0, "real", 1),
        ("r3", 0, "real", 0),
        ("r4", 0, "real", 0),
        ("f1", 1, "sd14", 1),
        ("f2", 1, "sd14", 0),
        ("f3", 1, "sd14", 1),
        ("f4", 1, "sd14", 1),
    ],
)


def kakao_group(ids):
    """kakao 4장: real 2장 모두 오탐, fake 2장 모두 미탐 → 정확도 0, 오탐률 1, 미탐률 1."""
    (a, b), (c, d) = ids[:2], ids[2:]
    return ("kakao", "kakao", [(a, 0, "real", 1), (b, 0, "real", 1), (c, 1, "sd14", 0), (d, 1, "sd14", 0)])


def test_kakao_drop_linked():
    """A11: 모두 연결 → 같은 origin_id의 genimage original(r1, r2, f1, f2)만 기준."""
    table = make_predictions([GENIMAGE_ORIGINAL, kakao_group(["r1", "r2", "f1", "f2"])])
    r = drop_row(compute_drop(table), "kakao", "kakao")
    # 기준 r1(맞음) r2(오탐) f1(맞음) f2(미탐): 정확도 2/4, 오탐률 1/2, 미탐률 1/2
    assert r["baseline"] == BASELINE_LINKED
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (2, 2)
    assert r["accuracy_drop_pp"] == pytest.approx(50.0)  # 50 − 0
    assert r["false_positive_rate_increase_pp"] == pytest.approx(50.0)  # 100 − 50
    assert r["false_negative_rate_increase_pp"] == pytest.approx(50.0)  # 100 − 50


def test_kakao_drop_unlinked_uses_all_genimage_original():
    """A11: 모두 미연결 → genimage original 전체(8장)가 기준."""
    table = make_predictions([GENIMAGE_ORIGINAL, kakao_group(["k1", "k2", "k3", "k4"])])
    r = drop_row(compute_drop(table), "kakao", "kakao")
    assert r["baseline"] == BASELINE_ALL
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (4, 4)
    assert r["accuracy_drop_pp"] == pytest.approx(75.0)  # 75 − 0
    assert r["false_positive_rate_increase_pp"] == pytest.approx(75.0)  # 100 − 25
    assert r["false_negative_rate_increase_pp"] == pytest.approx(75.0)  # 100 − 25


def test_kakao_drop_partially_linked_raises():
    """A11: 일부만 연결 → 에러."""
    table = make_predictions([GENIMAGE_ORIGINAL, kakao_group(["r1", "r2", "k3", "k4"])])
    with pytest.raises(KakaoLinkError):
        compute_drop(table)


# ---------------------------------------------------------------- predictions.csv


def test_predictions_roundtrip_exact(tmp_path):
    """A6: predictions.csv를 쓰고 다시 읽으면 logit·score까지 값이 정확히 같다."""
    rows = pd.DataFrame(
        {
            "path": ["a/1.png", "a/2.jpg", "b/3.png", "b/4.png", "c/5.png"],
            "label": ["real", "fake", "fake", "real", "fake"],
            "source": ["genimage"] * 5,
            "generator": ["real", "sd14", "mj", "real", "sd14"],
            "condition": ["original", "jpeg75", "jpeg75", "original", "resize50"],
            "origin_id": ["001", "002", "003", "004", "005"],  # 앞의 0이 사라지면 안 된다
            "gray": [False, True, False, True, False],  # 보완 2 (manifest의 bool gray)
        },
        index=[10, 11, 25, 30, 42],
    )
    logits = [0.1 + 0.2, -1e-300, 0.0, -745.3, 123456.78901234568]
    pred = build_predictions(rows, logits)
    assert list(pred["row_index"]) == [10, 11, 25, 30, 42]
    assert list(pred["label"]) == [0, 1, 1, 0, 1]
    assert list(pred["pred"]) == [1, 0, 1, 0, 1]  # 로짓 0은 fake
    assert pred.loc[2, "score"] == 0.5

    path = tmp_path / "predictions.csv"
    write_predictions(path, pred)
    back = read_predictions(path)
    pd.testing.assert_frame_equal(back, pred, check_exact=True)
    assert list(back["origin_id"]) == ["001", "002", "003", "004", "005"]


# ---------------------------------------------------------------- 시작 로그 확장자 분포 (처리 방식 1-5)


def manifest_rows(spec):
    """spec: [(source, condition, label, ext), ...]"""
    return pd.DataFrame(
        [
            {"path": f"images/{s}/{c}/{lab}/{i}.{ext}", "source": s, "condition": c, "label": lab}
            for i, (s, c, lab, ext) in enumerate(spec)
        ]
    )


def test_format_bias_warning_when_real_fake_ext_differ():
    rows = manifest_rows(
        [
            ("genimage", "original", "real", "jpg"),
            ("genimage", "original", "fake", "png"),
            ("genimage", "jpeg75", "real", "jpg"),
            ("genimage", "jpeg75", "fake", "jpg"),
            ("latest", "original", "fake", "png"),  # 한 라벨만 있으면 비교하지 않는다
        ]
    )
    warnings = format_bias_warnings(rows)
    assert len(warnings) == 1
    assert "(genimage, original)" in warnings[0]
