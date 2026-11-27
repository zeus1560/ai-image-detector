"""SPEC-03 확정 후 보완 2 A 테스트: gray 분리, gray_real 그룹, 생성기별 전체 지표, train_generator, 생성기별 drop, origin_id 중복 검사.

기대값은 모두 손으로 계산한 숫자다 (코드로 다시 계산하지 않는다).
로짓은 make_predictions 규칙대로 행 순서 i에 따라 (1+0.1i)×(pred면 +1, 아니면 −1)이다.
"""

import pandas as pd
import pytest

from conftest import make_row, valid_rows
from test_eval_report import EXPECTED_CONDITION, HAND_TABLE, make_predictions
from src.model.dataset import ManifestDataset
from src.model.eval_report import (
    BASELINE_ALL,
    BASELINE_LINKED,
    BASELINE_SAME_SOURCE,
    LEGACY_PRED_COLUMNS,
    DuplicateOriginError,
    PredictionsFormatError,
    check_unique_origin_per_condition,
    compute_drop,
    read_predictions,
    summarize,
    write_predictions,
)

# ---------------------------------------------------------------- 손 표 G
# 실제 컬러·흑백, 생성기 2개(g1, g2), 흑백 가짜(BG1) 포함. 튜플: (origin_id, label, generator, pred, gray)
G_ORIGINAL = (
    "genimage",
    "original",
    [
        ("R1", 0, "real", 0, False),  # i0  −1.0
        ("R2", 0, "real", 0, False),  # i1  −1.1
        ("R3", 0, "real", 1, False),  # i2  +1.2 (오탐)
        ("RG1", 0, "real", 1, True),  # i3  +1.3 흑백 실제 (오탐)
        ("RG2", 0, "real", 0, True),  # i4  −1.4 흑백 실제
        ("A1", 1, "g1", 1, False),  # i5  +1.5
        ("A2", 1, "g1", 1, False),  # i6  +1.6
        ("A3", 1, "g1", 0, False),  # i7  −1.7 (미탐)
        ("B1", 1, "g2", 1, False),  # i8  +1.8
        ("B2", 1, "g2", 0, False),  # i9  −1.9 (미탐)
        ("BG1", 1, "g2", 0, True),  # i10 −2.0 흑백 가짜 (어디에도 안 들어감)
    ],
)
G_JPEG75 = (
    "genimage",
    "jpeg75",
    [
        ("R1", 0, "real", 1, False),  # +2.1 (오탐)
        ("R2", 0, "real", 0, False),  # −2.2
        ("R3", 0, "real", 1, False),  # +2.3 (오탐)
        ("A1", 1, "g1", 0, False),  # 미탐
        ("A2", 1, "g1", 0, False),  # 미탐
        ("A3", 1, "g1", 0, False),  # 미탐
        ("B1", 1, "g2", 1, False),
        ("B2", 1, "g2", 0, False),  # 미탐
        ("BG1", 1, "g2", 1, True),  # 흑백 가짜
        # 흑백 실제 없음 → gray_real n_real=0
    ],
)
LATEST = ("latest", "original", [("L1", 1, "flux", 1, False)])  # 실제 없음 → gray_real 행 없음

G = make_predictions([G_ORIGINAL, G_JPEG75, LATEST])


def one(table, group, source, condition, generator=None):
    sel = (table["group"] == group) & (table["source"] == source) & (table["condition"] == condition)
    if generator is not None:
        sel &= table["generator"] == generator
    r = table[sel]
    assert len(r) == 1, r
    return r.iloc[0]


# ---------------------------------------------------------------- predictions.csv gray 열


def test_legacy_predictions_read_gray_all_false(tmp_path, capsys):
    """예전 형식(gray 없음)은 gray 전부 False, 로그를 낸다. condition 행 값이 예전과 같다 (B 회귀의 로컬 대응)."""
    path = tmp_path / "predictions.csv"
    HAND_TABLE[LEGACY_PRED_COLUMNS].to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    back = read_predictions(path)
    assert "gray 열 없는 예전 파일" in capsys.readouterr().out
    assert back["gray"].dtype == bool and not back["gray"].any()

    cond = summarize(back)
    cond = cond[cond["group"] == "condition"]
    assert len(cond) == len(EXPECTED_CONDITION)
    for _, r in cond.iterrows():
        n_real, n_fake, n_fp, n_fn, acc, fpr, fnr = EXPECTED_CONDITION[(r["source"], r["condition"])]
        assert (r["n_real"], r["n_fake"], r["n_false_positive"], r["n_false_negative"]) == (n_real, n_fake, n_fp, n_fn)
        assert r["accuracy"] == pytest.approx(acc)


def test_new_format_roundtrip_keeps_gray(tmp_path):
    path = tmp_path / "predictions.csv"
    write_predictions(path, G)
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0].endswith(",gray")
    assert text[4].endswith(",True")  # RG1 (i3)
    back = read_predictions(path)
    pd.testing.assert_frame_equal(back, G, check_exact=True)


def _write_with_gray(tmp_path, values):
    path = tmp_path / "predictions.csv"
    write_predictions(path, G)
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, v in values.items():  # i: 데이터 행 번호 (0부터)
        head, _, _ = lines[i + 1].rpartition(",")
        lines[i + 1] = f"{head},{v}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_blank_gray_is_error_first(tmp_path):
    """빈 값은 에러다. 허용값 밖 값이 함께 있어도 빈 값이 먼저 보고된다."""
    path = _write_with_gray(tmp_path, {2: "", 5: "true"})
    with pytest.raises(PredictionsFormatError) as exc:
        read_predictions(path)
    assert "빈 값" in str(exc.value) and "[4]" in str(exc.value)  # 데이터 행 2 → 4번 줄


@pytest.mark.parametrize("value", ["true", "TRUE", "1", " True", "nan"])
def test_gray_not_exact_true_false_is_error(tmp_path, value):
    path = _write_with_gray(tmp_path, {5: value})
    with pytest.raises(PredictionsFormatError) as exc:
        read_predictions(path)
    assert f"7번 줄='{value}'" in str(exc.value)


def test_other_header_is_error(tmp_path):
    path = tmp_path / "predictions.csv"
    G.drop(columns=["origin_id"]).to_csv(path, index=False)
    with pytest.raises(PredictionsFormatError):
        read_predictions(path)


# ---------------------------------------------------------------- summary: condition 행


def test_condition_rows_use_only_gray_false():
    s = summarize(G)
    r = one(s, "condition", "genimage", "original")
    # 실제 R1 R2 R3(오탐) / 가짜 A1 A2 A3(미탐) B1 B2(미탐). 흑백 RG1 RG2 BG1 제외
    assert (r["n_real"], r["n_fake"], r["n_false_positive"], r["n_false_negative"]) == (3, 5, 1, 2)
    assert r["accuracy"] == pytest.approx(5 / 8)
    assert r["false_positive_rate"] == pytest.approx(1 / 3)
    assert r["false_negative_rate"] == pytest.approx(2 / 5)
    # AUC: 가짜 +1.5 +1.6 +1.8은 실제 3장보다 모두 큼(9), −1.7 −1.9는 모두 작음(0) → 9/15
    assert r["auc"] == pytest.approx(3 / 5)
    assert r["generator"] is None and r["train_generator"] is None

    r = one(s, "condition", "genimage", "jpeg75")
    assert (r["n_real"], r["n_fake"], r["n_false_positive"], r["n_false_negative"]) == (3, 5, 2, 4)
    assert r["accuracy"] == pytest.approx(2 / 8)
    assert r["false_positive_rate"] == pytest.approx(2 / 3)
    assert r["false_negative_rate"] == pytest.approx(4 / 5)


# ---------------------------------------------------------------- summary: gray_real 행


def test_gray_real_rows():
    s = summarize(G)
    gray = s[s["group"] == "gray_real"]
    # 실제 행이 있는 (source, condition)마다 한 행. latest는 실제가 없어 행이 없다
    assert set(zip(gray["source"], gray["condition"])) == {("genimage", "original"), ("genimage", "jpeg75")}

    r = one(s, "gray_real", "genimage", "original")
    assert (r["n_real"], r["n_false_positive"]) == (2, 1)  # RG1 오탐, RG2 맞음
    assert r["false_positive_rate"] == pytest.approx(1 / 2)
    assert round(r["false_positive_rate_ci_low"], 4) == 0.0945  # Wilson(1, 2) 손 계산
    assert round(r["false_positive_rate_ci_high"], 4) == 0.9055
    # 오탐률과 그 구간만 채운다
    for col in ("n_fake", "n_false_negative", "accuracy", "false_negative_rate", "ap", "auc"):
        assert r[col] is None

    r = one(s, "gray_real", "genimage", "jpeg75")
    assert r["n_real"] == 0 and r["n_false_positive"] == 0
    assert r["false_positive_rate"] is None
    assert r["false_positive_rate_ci_low"] is None and r["false_positive_rate_ci_high"] is None


# ---------------------------------------------------------------- summary: generator 행


# (condition, generator): n_real, n_fake, 오탐 수, 미탐 수, 정확도, 오탐률, 미탐률 (손 계산)
EXPECTED_G_GENERATOR = {
    ("original", "g1"): (3, 3, 1, 1, 4 / 6, 1 / 3, 1 / 3),
    ("original", "g2"): (3, 2, 1, 1, 3 / 5, 1 / 3, 1 / 2),  # BG1(흑백 가짜) 제외
    ("jpeg75", "g1"): (3, 3, 2, 3, 1 / 6, 2 / 3, 3 / 3),
    ("jpeg75", "g2"): (3, 2, 2, 1, 2 / 5, 2 / 3, 1 / 2),
}


def test_generator_rows_shared_real_and_full_metrics():
    s = summarize(G)
    for (condition, generator), expected in EXPECTED_G_GENERATOR.items():
        r = one(s, "generator", "genimage", condition, generator)
        n_real, n_fake, n_fp, n_fn, acc, fpr, fnr = expected
        assert (r["n_real"], r["n_fake"], r["n_false_positive"], r["n_false_negative"]) == (n_real, n_fake, n_fp, n_fn)
        assert r["accuracy"] == pytest.approx(acc)
        assert r["false_positive_rate"] == pytest.approx(fpr)
        assert r["false_negative_rate"] == pytest.approx(fnr)
        # n_real은 공유 실제(gray==False) 장수와 같다
        assert r["n_real"] == one(s, "condition", "genimage", condition)["n_real"]

    # AP·AUC (original). 실제 −1.0 −1.1 +1.2
    g1 = one(s, "generator", "genimage", "original", "g1")  # 가짜 +1.5 +1.6 −1.7
    assert g1["auc"] == pytest.approx(6 / 9)
    assert g1["ap"] == pytest.approx(1 / 3 + 1 / 3 + (1 / 3) * (3 / 6))
    g2 = one(s, "generator", "genimage", "original", "g2")  # 가짜 +1.8 −1.9 (−2.0 흑백 제외)
    assert g2["auc"] == pytest.approx(3 / 6)
    assert g2["ap"] == pytest.approx(1 / 2 + (1 / 2) * (2 / 5))

    # 생성기 행끼리 오탐률이 같다
    for condition in ("original", "jpeg75"):
        gen = s[(s["group"] == "generator") & (s["source"] == "genimage") & (s["condition"] == condition)]
        assert len(set(gen["false_positive_rate"])) == 1

    # 실제가 없는 latest의 생성기 행: n_real 0, 오탐률 빈칸
    r = one(s, "generator", "latest", "original", "flux")
    assert (r["n_real"], r["n_fake"]) == (0, 1) and r["false_positive_rate"] is None


def test_summary_group_order():
    groups = list(summarize(G)["group"])
    assert groups == sorted(groups, key=["condition", "gray_real", "generator"].index)


# ---------------------------------------------------------------- train_generator 열


def test_train_generator_marks():
    s = summarize(G, ["g1"])
    d = compute_drop(G, ["g1"])
    for table in (s, d):
        gen = table[table["group"] == "generator"]
        marks = dict(zip(zip(gen["condition"], gen["generator"]), gen["train_generator"]))
        assert marks[("jpeg75", "g1")] == "yes" and marks[("jpeg75", "g2")] == "no"
        assert table.loc[table["group"] != "generator", "train_generator"].isna().all()
    assert one(s, "generator", "genimage", "original", "g1")["train_generator"] == "yes"
    assert one(s, "generator", "latest", "original", "flux")["train_generator"] == "no"

    # 키가 없으면(CIFAKE) 모든 행 빈칸
    for table in (summarize(G), compute_drop(G)):
        assert table["train_generator"].isna().all()


# ---------------------------------------------------------------- drop: 생성기별 (같은 source)


def test_drop_generator_rows_same_source():
    d = compute_drop(G)
    assert set(d["group"]) == {"condition", "generator"}  # gray_real은 drop 없음

    r = one(d, "condition", "genimage", "jpeg75")
    assert r["baseline"] == BASELINE_SAME_SOURCE
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (3, 5)  # 기준도 gray==False
    assert r["accuracy_drop_pp"] == pytest.approx(37.5)  # 62.5 − 25
    assert r["false_positive_rate_increase_pp"] == pytest.approx(100 / 3)  # 66.67 − 33.33
    assert r["false_negative_rate_increase_pp"] == pytest.approx(40.0)  # 80 − 40

    r = one(d, "generator", "genimage", "jpeg75", "g1")
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (3, 3)
    assert r["accuracy_drop_pp"] == pytest.approx(50.0)  # 66.67 − 16.67
    assert r["false_positive_rate_increase_pp"] == pytest.approx(100 / 3)
    assert r["false_negative_rate_increase_pp"] == pytest.approx(200 / 3)  # 100 − 33.33

    r = one(d, "generator", "genimage", "jpeg75", "g2")
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (3, 2)  # BG1 제외
    assert r["accuracy_drop_pp"] == pytest.approx(20.0)  # 60 − 40
    assert r["false_positive_rate_increase_pp"] == pytest.approx(100 / 3)
    assert r["false_negative_rate_increase_pp"] == pytest.approx(0.0)  # 50 − 50


# ---------------------------------------------------------------- drop: 생성기별 (kakao 연결·전체)

# genimage original: 실제 r1(맞음) r2(오탐) r3 r4(맞음) / sd14 f1(맞음) f2(미탐) / mj m1 m2(미탐)
K_GENIMAGE = (
    "genimage",
    "original",
    [
        ("r1", 0, "real", 0),
        ("r2", 0, "real", 1),
        ("r3", 0, "real", 0),
        ("r4", 0, "real", 0),
        ("f1", 1, "sd14", 1),
        ("f2", 1, "sd14", 0),
        ("m1", 1, "mj", 0),
        ("m2", 1, "mj", 0),
    ],
)


def k_kakao(ids):
    """kakao: 실제 2장 모두 오탐, sd14 1장 미탐, mj 1장 맞음."""
    a, b, c, d = ids
    return ("kakao", "kakao", [(a, 0, "real", 1), (b, 0, "real", 1), (c, 1, "sd14", 0), (d, 1, "mj", 1)])


def test_kakao_drop_generator_linked():
    d = compute_drop(make_predictions([K_GENIMAGE, k_kakao(["r1", "r2", "f1", "m1"])]))
    # kakao sd14 행: 실제 2 + f1 → 정확도 0, 오탐률 1, 미탐률 1
    # 기준(연결): r1 r2 + f1 → 정확도 2/3, 오탐률 1/2, 미탐률 0
    r = one(d, "generator", "kakao", "kakao", "sd14")
    assert r["baseline"] == BASELINE_LINKED
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (2, 1)
    assert r["accuracy_drop_pp"] == pytest.approx(200 / 3)
    assert r["false_positive_rate_increase_pp"] == pytest.approx(50.0)
    assert r["false_negative_rate_increase_pp"] == pytest.approx(100.0)
    # kakao mj 행: 실제 2 + m1(맞음) → 정확도 1/3, 미탐률 0
    # 기준(연결): r1 r2 + m1(미탐) → 정확도 1/3, 오탐률 1/2, 미탐률 1
    r = one(d, "generator", "kakao", "kakao", "mj")
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (2, 1)
    assert r["accuracy_drop_pp"] == pytest.approx(0.0)
    assert r["false_positive_rate_increase_pp"] == pytest.approx(50.0)
    assert r["false_negative_rate_increase_pp"] == pytest.approx(-100.0)


def test_kakao_drop_generator_all():
    d = compute_drop(make_predictions([K_GENIMAGE, k_kakao(["k1", "k2", "k3", "k4"])]))
    # 기준(전체) sd14: 실제 4(오탐 1) + f1 f2(미탐 1) → 정확도 4/6, 오탐률 1/4, 미탐률 1/2
    r = one(d, "generator", "kakao", "kakao", "sd14")
    assert r["baseline"] == BASELINE_ALL
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (4, 2)
    assert r["accuracy_drop_pp"] == pytest.approx(200 / 3)  # 66.67 − 0
    assert r["false_positive_rate_increase_pp"] == pytest.approx(75.0)  # 100 − 25
    assert r["false_negative_rate_increase_pp"] == pytest.approx(50.0)  # 100 − 50
    # 기준(전체) mj: 실제 4(오탐 1) + m1 m2(둘 다 미탐) → 정확도 3/6, 미탐률 1
    r = one(d, "generator", "kakao", "kakao", "mj")
    assert (r["baseline_n_real"], r["baseline_n_fake"]) == (4, 2)
    assert r["accuracy_drop_pp"] == pytest.approx(50 - 100 / 3)  # 50 − 33.33
    assert r["false_positive_rate_increase_pp"] == pytest.approx(75.0)
    assert r["false_negative_rate_increase_pp"] == pytest.approx(-100.0)  # 0 − 100


# ---------------------------------------------------------------- origin_id 중복 검사 (보완 2의 5)


def rows_of(spec):
    return pd.DataFrame([{"source": s, "condition": c, "origin_id": o} for s, c, o in spec])


def test_duplicate_origin_in_same_source_condition_is_error():
    rows = rows_of(
        [
            ("genimage", "original", "r1"),
            ("genimage", "original", "r1"),  # 공유 실제가 생성기마다 따로 등록된 경우
            ("genimage", "original", "f1"),
        ]
    )
    with pytest.raises(DuplicateOriginError) as exc:
        check_unique_origin_per_condition(rows)
    assert "(genimage, original)" in str(exc.value) and "r1" in str(exc.value)


def test_same_origin_across_conditions_and_sources_is_not_duplicate():
    """검사 단위는 (source, condition)이다. 원본·변형본과 kakao 연결은 같은 origin_id여도 통과한다."""
    rows = rows_of(
        [
            ("genimage", "original", "r1"),
            ("genimage", "jpeg75", "r1"),
            ("kakao", "kakao", "r1"),
        ]
    )
    assert check_unique_origin_per_condition(rows) == "(source, condition)별 origin_id 중복 0개 (평가 대상 3행)"


def test_duplicate_check_uses_only_evaluated_split(build):
    """검사 대상은 평가하는 split으로 거른 행이다 (evaluate.py와 같은 경로: ManifestDataset(split=...).df).

    train split에만 중복이 있으면 test 평가는 통과하고, train 행을 대상으로 하면 에러가 난다.
    """
    rows = valid_rows()
    dup = make_row("r0", "real", "train")  # r0(train, original)을 한 번 더 등록
    dup["path"] = "images/cifake/original/real/r0_copy.png"
    rows.append(dup)
    manifest_path, data_root = build(rows)

    test_rows = ManifestDataset(manifest_path, data_root, split="test").df
    assert set(test_rows["split"]) == {"test"}
    check_unique_origin_per_condition(test_rows)

    train_rows = ManifestDataset(manifest_path, data_root, split="train").df
    with pytest.raises(DuplicateOriginError) as exc:
        check_unique_origin_per_condition(train_rows)
    assert "(cifake, original)" in str(exc.value) and "r0" in str(exc.value)
