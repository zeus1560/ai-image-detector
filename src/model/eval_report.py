"""조건별 평가 결과 표 (SPEC-03 처리 방식 1-5, 2, 3, 4, 출력 2).

torch를 import하지 않는다. 지표 계산은 metrics.py를 재사용한다.

- build_predictions / write_predictions / read_predictions: predictions.csv
- wilson_interval: 비율의 Wilson 95% 신뢰구간
- summarize: summary.csv (그룹별 지표)
- compute_drop: drop.csv (원본 대비 하락, %p)
- extension_table / format_bias_warnings: 시작 로그의 확장자 분포와 포맷 편향 경고
- check_unique_origin_per_condition: 평가 시작 시 origin_id 중복 검사 (보완 2의 5)

확정 후 보완 2: predictions.csv에 gray 열, summary 그룹 condition / gray_real / generator,
summary·drop에 train_generator 열, drop도 생성기별.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import LABEL_TO_INDEX
from .metrics import average_precision, label_metrics, roc_auc

PRED_COLUMNS = [
    "row_index",
    "path",
    "label",  # 0=real, 1=fake
    "source",
    "generator",
    "condition",
    "origin_id",
    "logit",
    "score",  # sigmoid(logit) = fake일 확률
    "pred",  # 1(fake) if logit >= 0 else 0
    "gray",  # True/False (보완 2, manifest의 gray. 맨 끝)
]
# 보완 2 이전 형식 (gray 없음). 읽으면 gray를 전부 False로 채운다
LEGACY_PRED_COLUMNS = PRED_COLUMNS[:-1]
_INT_COLUMNS = ("row_index", "label", "pred")
_FLOAT_COLUMNS = ("logit", "score")
_STR_COLUMNS = ("path", "source", "generator", "condition", "origin_id")
GRAY_VALUES = {"True": True, "False": False}

# 실수를 다시 읽었을 때 같은 값이 되도록 유효숫자 17자리로 쓴다 (float64 왕복 보장)
FLOAT_FORMAT = "%.17g"

# 표준정규분포 0.975 분위수 (Wilson 95% 구간)
Z_95 = 1.959963984540054

ORIGINAL = "original"
REAL_GENERATOR = "real"

# drop.csv의 kakao 기준 (처리 방식 4)
KAKAO_SOURCE = "kakao"
KAKAO_BASELINE_SOURCE = "genimage"

BASELINE_SAME_SOURCE = "same_source_original"
BASELINE_LINKED = "genimage_original_linked"  # 같은 origin_id의 genimage original 행만
BASELINE_ALL = "genimage_original_all"  # genimage original 전체 (서로 다른 이미지 간 비교)
BASELINE_NONE = "none"

BASELINE_DESCRIPTIONS = {
    BASELINE_SAME_SOURCE: "같은 source의 original",
    BASELINE_LINKED: "genimage original 중 같은 origin_id 행 (같은 사진의 전·후 비교)",
    BASELINE_ALL: "genimage original 전체 (서로 다른 이미지 간 비교라는 한계 있음)",
    BASELINE_NONE: "기준 original 없음 (빈칸)",
}

GROUP_CONDITION = "condition"  # (source, condition)의 gray==False 행 (대표 지표)
GROUP_GRAY_REAL = "gray_real"  # (source, condition)의 실제 gray==True 행 (오탐률만)
GROUP_GENERATOR = "generator"  # 공유 실제 gray==False + 그 생성기 가짜 gray==False (전체 지표)

SUMMARY_COLUMNS = [
    "group",  # GROUP_CONDITION / GROUP_GRAY_REAL / GROUP_GENERATOR
    "source",
    "condition",
    "generator",  # group=generator가 아니면 빈칸
    "train_generator",  # generator 행: yes/no (학습 기록에 train_generators가 있을 때만), 그 밖은 빈칸
    "n_real",
    "n_fake",
    "n_false_positive",
    "n_false_negative",
    "accuracy",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "false_positive_rate",
    "false_positive_rate_ci_low",
    "false_positive_rate_ci_high",
    "false_negative_rate",
    "false_negative_rate_ci_low",
    "false_negative_rate_ci_high",
    "ap",
    "auc",
]

# 하락 = original − 조건 (정확도, AP, AUC) / 증가 = 조건 − original (오탐률, 미탐률). 단위 %p
DROP_METRICS = [
    ("accuracy", "accuracy_drop_pp", -1),
    ("false_positive_rate", "false_positive_rate_increase_pp", +1),
    ("false_negative_rate", "false_negative_rate_increase_pp", +1),
    ("ap", "ap_drop_pp", -1),
    ("auc", "auc_drop_pp", -1),
]

DROP_COLUMNS = [
    "group",  # GROUP_CONDITION / GROUP_GENERATOR (gray_real은 drop을 계산하지 않음)
    "source",
    "condition",
    "generator",  # group=generator가 아니면 빈칸
    "train_generator",
    "baseline",  # 기준 방식 (BASELINE_* 값)
    "baseline_n_real",
    "baseline_n_fake",
    *[col for _, col, _ in DROP_METRICS],
]


class PredictionsFormatError(ValueError):
    """predictions.csv 형식이 열 정의와 맞지 않을 때 나는 에러."""


class KakaoLinkError(ValueError):
    """kakao 행의 origin_id가 genimage original과 일부만 연결될 때 나는 에러 (manifest 오류)."""


class DuplicateOriginError(ValueError):
    """평가 대상 행의 (source, condition) 안에서 같은 origin_id가 두 번 이상 나올 때 나는 에러."""


# ---------------------------------------------------------------- predictions.csv


def sigmoid(logits) -> np.ndarray:
    """수치적으로 안정한 sigmoid (큰 음수 로짓에서 overflow 경고 없음)."""
    x = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -x))


def build_predictions(rows: pd.DataFrame, logits) -> pd.DataFrame:
    """manifest 행(평가 순서 그대로)과 로짓으로 predictions 표를 만든다.

    Args:
        rows: ManifestDataset.df (원래 행 인덱스 유지). label은 "real"/"fake" 문자열,
              gray는 bool (SPEC-01a v3의 load_manifest가 채움)
        logits: rows와 같은 순서의 로짓 (fake 쪽)
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or len(logits) != len(rows):
        raise ValueError(f"로짓 개수가 행 수와 다릅니다: {logits.shape} vs {len(rows)}")
    if not np.isfinite(logits).all():
        raise ValueError("로짓에 NaN 또는 무한대가 있습니다")

    out = pd.DataFrame(
        {
            "row_index": np.asarray(rows.index, dtype=np.int64),
            "path": rows["path"].to_numpy(),
            "label": rows["label"].map(LABEL_TO_INDEX).to_numpy(),
            "source": rows["source"].to_numpy(),
            "generator": rows["generator"].to_numpy(),
            "condition": rows["condition"].to_numpy(),
            "origin_id": rows["origin_id"].to_numpy(),
            "logit": logits,
            "score": sigmoid(logits),
            # 로짓 0 = 점수 0.5. 같으면 fake (train.py evaluate와 같은 규칙)
            "pred": (logits >= 0).astype(np.int64),
            "gray": rows["gray"].astype(bool).to_numpy(),
        }
    )
    if out["label"].isna().any():
        raise ValueError("label에 real/fake 이외의 값이 있습니다")
    out["label"] = out["label"].astype(np.int64)
    return out[PRED_COLUMNS]


def write_predictions(path, predictions: pd.DataFrame) -> None:
    """gray(bool)는 문자열 True / False로 쓰인다."""
    if list(predictions.columns) != PRED_COLUMNS:
        raise PredictionsFormatError(f"predictions 열이 다릅니다: {list(predictions.columns)}")
    predictions.to_csv(path, index=False, float_format=FLOAT_FORMAT, lineterminator="\n", encoding="utf-8")


def _parse_gray(values: pd.Series, path) -> np.ndarray:
    """gray 문자열을 bool로. 빈 값을 먼저 에러로 보고, 그 다음 정확히 True/False가 아닌 값을 본다."""
    blank = values.str.strip() == ""
    if blank.any():
        lines = [int(i) + 2 for i in values[blank].index[:10]]
        raise PredictionsFormatError(f"predictions.csv의 gray에 빈 값 {int(blank.sum())}개 ({path}). 줄 번호 예: {lines}")
    bad = ~values.isin(GRAY_VALUES)
    if bad.any():
        examples = [f"{int(i) + 2}번 줄='{v}'" for i, v in values[bad].head(10).items()]
        raise PredictionsFormatError(
            f"predictions.csv의 gray는 정확히 True/False여야 합니다 ({path}). 예: {', '.join(examples)}"
        )
    return values.map(GRAY_VALUES).to_numpy(dtype=bool)


def read_predictions(path) -> pd.DataFrame:
    """predictions.csv 읽기.

    헤더가 새 형식(PRED_COLUMNS)이면 gray 값을 검사하고, 예전 형식(gray 없음)이면 gray를 전부 False로 채운다.
    그 밖의 헤더는 에러.
    """
    # 실수 열은 문자열로 읽어 Python float()로 바꾼다.
    # pandas 버전마다 기본 실수 파서가 달라 마지막 자릿수가 바뀔 수 있기 때문이다 (float()는 %.17g를 정확히 되돌린다)
    dtypes = {
        **{c: "int64" for c in _INT_COLUMNS},
        **{c: str for c in (*_FLOAT_COLUMNS, *_STR_COLUMNS, "gray")},
    }
    df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, encoding="utf-8")
    columns = list(df.columns)
    if columns == PRED_COLUMNS:
        df["gray"] = _parse_gray(df["gray"], path)
    elif columns == LEGACY_PRED_COLUMNS:
        df["gray"] = False
        print(f"gray 열 없는 예전 파일 → gray 전부 False (행 {len(df)}개): {path}")
    else:
        raise PredictionsFormatError(f"predictions.csv 헤더가 다릅니다: {columns} (기대 {PRED_COLUMNS})")
    for c in _FLOAT_COLUMNS:
        df[c] = np.array([float(v) for v in df[c]], dtype=np.float64)
    return df


# ---------------------------------------------------------------- 평가 시작 검사 (보완 2의 5)


def check_unique_origin_per_condition(rows: pd.DataFrame) -> str:
    """(source, condition) 안에서 같은 origin_id가 두 번 이상 나오면 DuplicateOriginError.

    rows는 평가 대상 행이다: evaluate.py가 평가하는 split(과 source)으로 거른 ManifestDataset.df.
    다른 split의 행이나 다른 (source, condition)의 같은 origin_id(원본과 변형본, kakao 연결)는 중복이 아니다.
    공유 실제 사진이 생성기마다 따로 등록되면 생성기 행에서 같은 사진이 중복으로 세어지기 때문에 막는다.

    Returns:
        통과 로그 문장
    """
    dup = rows[rows.duplicated(["source", "condition", "origin_id"], keep=False)]
    if len(dup):
        groups = dup.groupby(["source", "condition"])["origin_id"].unique()
        details = [
            f"({source}, {condition}) origin_id {len(ids)}개 예: {', '.join(map(str, ids[:10]))}"
            for (source, condition), ids in groups.items()
        ]
        raise DuplicateOriginError(
            "평가 대상 행의 (source, condition) 안에서 origin_id가 중복됩니다\n- " + "\n- ".join(details)
        )
    return f"(source, condition)별 origin_id 중복 0개 (평가 대상 {len(rows)}행)"


# ---------------------------------------------------------------- 지표


def wilson_interval(successes: int, n: int, z: float = Z_95):
    """비율 successes/n의 Wilson 신뢰구간 (하한, 상한). n=0이면 (None, None)."""
    if n == 0:
        return None, None
    if not 0 <= successes <= n:
        raise ValueError(f"successes는 0~n이어야 합니다: {successes}/{n}")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # 0/n이면 하한, n/n이면 상한이 수식상 정확히 0, 1이다 (부동소수 오차로 0.9999…가 되지 않게)
    low = 0.0 if successes == 0 else center - margin
    high = 1.0 if successes == n else center + margin
    return low, high


def group_metrics(df: pd.DataFrame) -> dict:
    """predictions 행 묶음 하나의 개수, 지표, Wilson 구간, AP·AUC.

    분모가 0이거나 한 라벨만 있으면 해당 값은 None (metrics.py 규칙).
    """
    y_true = df["label"].to_numpy()
    y_pred = df["pred"].to_numpy()
    scores = df["score"].to_numpy()
    real = y_true == 0
    fake = y_true == 1
    n_real, n_fake = int(real.sum()), int(fake.sum())
    n_fp = int((y_pred[real] == 1).sum())
    n_fn = int((y_pred[fake] == 0).sum())
    n_correct = int((y_true == y_pred).sum())

    out = {"n_real": n_real, "n_fake": n_fake, "n_false_positive": n_fp, "n_false_negative": n_fn}
    out.update(label_metrics(y_true, y_pred))
    for name, k, n in (
        ("accuracy", n_correct, n_real + n_fake),
        ("false_positive_rate", n_fp, n_real),
        ("false_negative_rate", n_fn, n_fake),
    ):
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = wilson_interval(k, n)
    out["ap"] = average_precision(y_true, scores)
    out["auc"] = roc_auc(y_true, scores)
    return out


def _condition_order(conditions) -> list:
    """original을 맨 앞에, 나머지는 이름순."""
    conditions = sorted(set(conditions))
    return [c for c in conditions if c == ORIGINAL] + [c for c in conditions if c != ORIGINAL]


def _iter_condition_groups(predictions: pd.DataFrame):
    for source in sorted(predictions["source"].unique()):
        by_source = predictions[predictions["source"] == source]
        for condition in _condition_order(by_source["condition"]):
            yield source, condition, by_source[by_source["condition"] == condition]


def _color_rows(g: pd.DataFrame) -> pd.DataFrame:
    """대표 지표용 행: gray == False (흑백 실제·흑백 가짜 모두 뺀다)."""
    return g[~g["gray"]]


def _generator_rows(g: pd.DataFrame, generator: str) -> pd.DataFrame:
    """생성기 행용: 실제 gray==False 전체(공유 실제) + 그 생성기의 가짜 gray==False."""
    color = _color_rows(g)
    return color[(color["label"] == 0) | ((color["label"] == 1) & (color["generator"] == generator))]


def _fake_generators(g: pd.DataFrame) -> list:
    return sorted(g.loc[g["generator"] != REAL_GENERATOR, "generator"].unique())


def _train_generator_mark(generator, train_generators):
    """generator 행의 train_generator 값. train_generators가 None이면 빈칸."""
    if train_generators is None:
        return None
    return "yes" if generator in train_generators else "no"


def summarize(predictions: pd.DataFrame, train_generators=None) -> pd.DataFrame:
    """summary.csv 표 (처리 방식 3, 보완 2의 2·3).

    1. group="condition": (source, condition)의 gray==False 행. 정확도·오탐률·미탐률(+Wilson 구간)·AP·ROC-AUC
    2. group="gray_real": 실제 행이 있는 (source, condition)마다 실제 gray==True 행. n_real·오탐 수·오탐률(+Wilson)만
    3. group="generator": (source, condition, generator≠real)마다 공유 실제 gray==False + 그 생성기 가짜 gray==False. 전체 지표
    흑백 가짜는 어디에도 들어가지 않는다.
    train_generators: 학습 기록(run_dir/config.json)의 값. 있으면 generator 행에 yes/no, 없으면(None) 빈칸.
    """
    base = {"generator": None, "train_generator": None}
    rows = []
    for source, condition, g in _iter_condition_groups(predictions):
        rows.append({"group": GROUP_CONDITION, "source": source, "condition": condition, **base, **group_metrics(_color_rows(g))})

    for source, condition, g in _iter_condition_groups(predictions):
        if not (g["label"] == 0).any():
            continue
        gray_real = g[(g["label"] == 0) & g["gray"]]
        n_real = len(gray_real)
        n_fp = int((gray_real["pred"] == 1).sum())
        low, high = wilson_interval(n_fp, n_real)
        rows.append(
            {
                "group": GROUP_GRAY_REAL,
                "source": source,
                "condition": condition,
                **base,
                "n_real": n_real,
                "n_false_positive": n_fp,
                "false_positive_rate": None if n_real == 0 else n_fp / n_real,
                "false_positive_rate_ci_low": low,
                "false_positive_rate_ci_high": high,
            }
        )

    for source, condition, g in _iter_condition_groups(predictions):
        for generator in _fake_generators(g):
            rows.append(
                {
                    "group": GROUP_GENERATOR,
                    "source": source,
                    "condition": condition,
                    "generator": generator,
                    "train_generator": _train_generator_mark(generator, train_generators),
                    **group_metrics(_generator_rows(g, generator)),
                }
            )

    table = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return table.astype(object).where(table.notna(), None)


# ---------------------------------------------------------------- drop


def kakao_baseline(predictions: pd.DataFrame):
    """kakao 행의 비교 기준을 데이터로 자동 판정한다 (처리 방식 4).

    kakao 행의 origin_id가
    - 모두 genimage original에 있으면 BASELINE_LINKED: 그 origin_id의 genimage original 행만
    - 하나도 없으면 BASELINE_ALL: genimage original 전체
    - 일부만 있으면 KakaoLinkError (manifest가 잘못 만들어진 것)
    genimage original이 없으면 BASELINE_NONE.

    predictions는 한 split만 담으므로 여기의 genimage original은 곧 같은 split의 original이다.

    Returns:
        (기준 방식, 기준 행 DataFrame 또는 None)
    """
    kakao_ids = set(predictions.loc[predictions["source"] == KAKAO_SOURCE, "origin_id"])
    base = predictions[(predictions["source"] == KAKAO_BASELINE_SOURCE) & (predictions["condition"] == ORIGINAL)]
    if len(base) == 0:
        return BASELINE_NONE, None
    linked = kakao_ids & set(base["origin_id"])
    if linked == kakao_ids:
        return BASELINE_LINKED, base[base["origin_id"].isin(kakao_ids)]
    if not linked:
        return BASELINE_ALL, base
    raise KakaoLinkError(
        f"kakao origin_id {len(kakao_ids)}개 중 {len(linked)}개만 genimage original과 연결됩니다. "
        "모두 연결되거나 하나도 연결되지 않아야 합니다 (manifest를 확인하세요). "
        f"연결 안 된 예: {', '.join(sorted(kakao_ids - linked)[:10])}"
    )


def _pp_change(base, value, sign):
    """sign=+1: (조건 − original)×100, sign=−1: (original − 조건)×100. 한쪽이라도 None이면 None."""
    if base is None or value is None:
        return None
    return (value - base) * 100 * sign


def _drop_row(head: dict, mode, cond_rows: pd.DataFrame, base_rows) -> dict:
    row = {**head, "baseline": mode}
    cond = group_metrics(cond_rows)
    base = group_metrics(base_rows) if base_rows is not None else None
    row["baseline_n_real"] = base["n_real"] if base else None
    row["baseline_n_fake"] = base["n_fake"] if base else None
    for metric, col, sign in DROP_METRICS:
        row[col] = _pp_change(base[metric], cond[metric], sign) if base else None
    return row


def compute_drop(predictions: pd.DataFrame, train_generators=None) -> pd.DataFrame:
    """drop.csv 표: original 대비 하락 (단위 %p, 처리 방식 4, 보완 2의 4).

    - 같은 source의 condition≠original 그룹마다 그 source의 original과 비교한다
    - kakao source는 genimage original을 기준으로 한다 (kakao_baseline으로 자동 판정, 규칙은 보완 2 이전과 같음)
    - 그 밖에 기준 original이 없으면 값을 빈칸으로 둔다
    - group="condition": 조건·기준 모두 gray==False 행
    - group="generator": 조건·기준 모두 "실제 gray==False + 같은 생성기 가짜 gray==False" 행
    - gray_real은 계산하지 않는다
    """
    cond_out, gen_out = [], []
    for source, condition, g in _iter_condition_groups(predictions):
        if condition == ORIGINAL:
            continue
        if source == KAKAO_SOURCE:
            mode, base_rows = kakao_baseline(predictions)
        else:
            base_rows = predictions[(predictions["source"] == source) & (predictions["condition"] == ORIGINAL)]
            mode = BASELINE_SAME_SOURCE if len(base_rows) else BASELINE_NONE
            if not len(base_rows):
                base_rows = None

        head = {"source": source, "condition": condition, "generator": None, "train_generator": None}
        cond_out.append(
            _drop_row(
                {"group": GROUP_CONDITION, **head},
                mode,
                _color_rows(g),
                _color_rows(base_rows) if base_rows is not None else None,
            )
        )
        for generator in _fake_generators(g):
            gen_head = {
                **head,
                "generator": generator,
                "train_generator": _train_generator_mark(generator, train_generators),
            }
            gen_out.append(
                _drop_row(
                    {"group": GROUP_GENERATOR, **gen_head},
                    mode,
                    _generator_rows(g, generator),
                    _generator_rows(base_rows, generator) if base_rows is not None else None,
                )
            )

    table = pd.DataFrame(cond_out + gen_out, columns=DROP_COLUMNS)
    return table.astype(object).where(table.notna(), None)


def write_table(path, table: pd.DataFrame) -> None:
    """summary.csv·drop.csv 쓰기. None은 빈칸."""
    table.to_csv(path, index=False, float_format=FLOAT_FORMAT, lineterminator="\n", encoding="utf-8")


# ---------------------------------------------------------------- 시작 로그: 확장자 분포


def extension_table(rows: pd.DataFrame) -> pd.DataFrame:
    """(source, condition, label)별 파일 확장자 개수표 (처리 방식 1-5)."""
    ext = rows["path"].map(lambda p: Path(p).suffix.lower() or "(없음)")
    return pd.crosstab([rows["source"], rows["condition"], rows["label"]], ext.rename("ext"))


def format_bias_warnings(rows: pd.DataFrame) -> list[str]:
    """같은 (source, condition)에서 real과 fake의 확장자 분포가 다르면 경고 문장을 돌려준다.

    한 라벨만 있는 그룹은 비교하지 않는다. 실행은 멈추지 않는다 (경고만).
    """
    table = extension_table(rows)
    warnings = []
    for (source, condition), g in table.groupby(level=[0, 1]):
        labels = g.index.get_level_values(2)
        if not {"real", "fake"} <= set(labels):
            continue
        ratio = g.div(g.sum(axis=1), axis=0).droplevel([0, 1])
        if not np.allclose(ratio.loc["real"].to_numpy(), ratio.loc["fake"].to_numpy()):
            def dist(label):
                r = ratio.loc[label]
                return ", ".join(f"{e} {v:.0%}" for e, v in r.items() if v > 0)

            warnings.append(
                f"포맷 편향 의심: ({source}, {condition})에서 real [{dist('real')}] / fake [{dist('fake')}]"
            )
    return warnings
