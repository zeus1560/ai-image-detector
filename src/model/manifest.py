"""manifest.csv 로더와 검증 (SPEC-01a).

모델 코드는 이 모듈을 통해서만 데이터 목록을 읽는다.
열 구성과 규칙은 docs/spec/data-contract.md를 따른다.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "path",
    "label",
    "source",
    "generator",
    "origin_id",
    "condition",
    "split",
    "width",
    "height",
    "format",
]

ALLOWED_VALUES = {
    "label": {"real", "fake"},
    "split": {"train", "val", "test"},
    "source": {"cifake", "genimage", "kakao", "latest"},
}

# kakao, latest 출처는 test 전용 (data-contract.md 규칙 4)
TEST_ONLY_SOURCES = {"kakao", "latest"}

# check_files=n 샘플 검사에 쓰는 고정 seed
FILE_CHECK_SEED = 0

# 에러 메시지에 보여줄 예시 최대 개수
MAX_EXAMPLES = 10


class ManifestValidationError(Exception):
    """manifest가 데이터 계약을 어겼을 때 나는 에러."""


class LeakageError(ManifestValidationError):
    """한 origin_id가 두 개 이상의 split에 있을 때 나는 에러 (데이터 누수)."""


def _csv_line(index: int) -> int:
    """DataFrame 행 인덱스를 CSV 파일의 줄 번호로 바꾼다 (헤더가 1번 줄)."""
    return int(index) + 2


def load_manifest(path, data_root) -> pd.DataFrame:
    """manifest CSV를 읽고 필수 열과 허용값(label, split, source)을 확인한다.

    모든 값은 문자열로 읽는다 (origin_id 등이 숫자로 바뀌지 않게).
    data_root는 df.attrs["data_root"]에 기록해 둔다.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestValidationError(
            f"manifest에 필수 열이 없습니다: {', '.join(missing)} (파일: {path})"
        )

    problems = []
    for col, allowed in ALLOWED_VALUES.items():
        bad = df[~df[col].isin(allowed)]
        if len(bad):
            examples = ", ".join(
                f"{_csv_line(i)}번 줄='{v}'" for i, v in bad[col].head(MAX_EXAMPLES).items()
            )
            problems.append(
                f"- 열 '{col}'에 허용되지 않은 값 {len(bad)}개 "
                f"(허용값: {sorted(allowed)}). 예: {examples}"
            )
    if problems:
        raise ManifestValidationError(
            f"manifest 허용값 검사 실패 (파일: {path}, 줄 번호는 헤더=1 기준)\n"
            + "\n".join(problems)
        )

    df.attrs["data_root"] = str(data_root)
    return df


def _file_exists(full_path: Path) -> bool:
    return full_path.is_file()


def validate(df: pd.DataFrame, data_root, check_files="all") -> None:
    """데이터 계약 규칙 1~4를 모두 검사하고, 위반을 모아 하나의 에러로 보고한다.

    1. 한 origin_id가 두 개 이상의 split에 존재 (누수)
    2. 한 origin_id 안에서 label이 다름
    3. source가 kakao 또는 latest인데 split이 test가 아님
    4. path가 가리키는 파일이 없음 (check_files="all" 전체, 정수 n이면 n개 샘플)

    1번 위반이 있으면 LeakageError, 없으면 ManifestValidationError를 낸다.
    """
    data_root = Path(data_root)
    violations = []
    log = []

    # 1. split 겹침
    splits_per_id = df.groupby("origin_id")["split"].nunique()
    overlap_ids = splits_per_id[splits_per_id > 1].index.tolist()
    if overlap_ids:
        examples = []
        for oid in overlap_ids[:MAX_EXAMPLES]:
            splits = sorted(df.loc[df["origin_id"] == oid, "split"].unique())
            examples.append(f"{oid}({'/'.join(splits)})")
        violations.append(
            f"[1] split 겹침 origin_id {len(overlap_ids)}개. 예: {', '.join(examples)}"
        )
    log.append(f"split 겹침 origin_id {len(overlap_ids)}개")

    # 2. origin_id 안에서 label 불일치
    labels_per_id = df.groupby("origin_id")["label"].nunique()
    mismatch_ids = labels_per_id[labels_per_id > 1].index.tolist()
    if mismatch_ids:
        violations.append(
            f"[2] origin_id 안에서 label이 다른 ID {len(mismatch_ids)}개. "
            f"예: {', '.join(mismatch_ids[:MAX_EXAMPLES])}"
        )
    log.append(f"label 불일치 origin_id {len(mismatch_ids)}개")

    # 3. test 전용 출처가 test 밖에 배정됨
    bad_split = df[df["source"].isin(TEST_ONLY_SOURCES) & (df["split"] != "test")]
    if len(bad_split):
        examples = ", ".join(
            f"{_csv_line(i)}번 줄({r.source}, split={r.split}, {r.path})"
            for i, r in bad_split.head(MAX_EXAMPLES).iterrows()
        )
        violations.append(
            f"[3] kakao/latest 출처인데 split이 test가 아닌 행 {len(bad_split)}개. 예: {examples}"
        )
    log.append(f"test 전용 출처의 test 외 배정 {len(bad_split)}개")

    # 4. 파일 존재
    if check_files == "all":
        targets = df
        log_files = f"파일 존재 검사: 전체 {len(df)}개"
    elif isinstance(check_files, int) and not isinstance(check_files, bool) and check_files > 0:
        n = min(check_files, len(df))
        rng = random.Random(FILE_CHECK_SEED)
        picked = sorted(rng.sample(range(len(df)), n))
        targets = df.iloc[picked]
        log_files = (
            f"파일 존재 검사: 샘플 검사 {n}개 / 전체 {len(df)}개 (seed={FILE_CHECK_SEED}). "
            f"나머지 {len(df) - n}개는 검사하지 않음"
        )
    else:
        raise ValueError(f"check_files는 'all' 또는 양의 정수여야 합니다: {check_files!r}")

    missing_paths = [p for p in targets["path"] if not _file_exists(data_root / p)]
    if missing_paths:
        violations.append(
            f"[4] 파일이 없는 path {len(missing_paths)}개 (data_root={data_root}). "
            f"예: {', '.join(missing_paths[:MAX_EXAMPLES])}"
        )
    log.append(f"{log_files}, 없는 파일 {len(missing_paths)}개")

    if violations:
        error_cls = LeakageError if overlap_ids else ManifestValidationError
        raise error_cls(
            f"manifest 검사 실패: 위반 {len(violations)}종류\n" + "\n".join(violations)
        )

    print(f"manifest 검사 통과 (행 {len(df)}개, origin_id {df['origin_id'].nunique()}개)")
    for line in log:
        print(f"  - {line}")


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """label × condition × split 장수표를 출력하고 반환한다.

    original 조건에서 label별 format·크기 분포를 출력하고,
    label 사이에 분포가 다르면 경고만 한다 (멈추지 않는다).
    """
    table = pd.crosstab(
        [df["label"], df["condition"]], df["split"], dropna=False
    )
    print("[장수표] label × condition × split")
    print(table.to_string())

    original = df[df["condition"] == "original"]
    if len(original) == 0:
        print("[original 분포] original 조건 행이 없습니다")
        return table

    original = original.assign(size=original["width"] + "x" + original["height"])
    for col in ("format", "size"):
        dist = pd.crosstab(original["label"], original[col], normalize="index")
        print(f"[original 분포] label별 {col} 비율")
        print(dist.round(3).to_string())
        if len(dist) > 1 and not (dist.nunique(axis=0) == 1).all():
            warnings.warn(
                f"original 조건에서 label별 {col} 분포가 다릅니다 (포맷 편향 가능성).",
                UserWarning,
                stacklevel=2,
            )
    return table


def filter(df: pd.DataFrame, source=None, condition=None, split=None) -> pd.DataFrame:
    """조건에 맞는 행만 반환한다. None인 조건은 거르지 않는다.

    원래 행 인덱스를 유지한다.
    """
    mask = pd.Series(True, index=df.index)
    if source is not None:
        mask &= df["source"] == source
    if condition is not None:
        mask &= df["condition"] == condition
    if split is not None:
        mask &= df["split"] == split
    return df[mask]
