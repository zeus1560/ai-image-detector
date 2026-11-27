"""manifest.csv 로더와 검증 (SPEC-01a).

모델 코드는 이 모듈을 통해서만 데이터 목록을 읽는다.
열 구성과 규칙은 docs/spec/data-contract.md를 따른다.
"""

from __future__ import annotations

import hashlib
import io
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
    "color_mode",
]

ALLOWED_VALUES = {
    "label": {"real", "fake"},
    "split": {"train", "val", "test"},
    "source": {"cifake", "genimage", "kakao", "latest"},
    # 변환 전 원본의 PIL 모드. 저장 파일은 모두 RGB (data-contract.md)
    "color_mode": {"RGB", "L", "CMYK"},
}

# 선택 열 gray (SPEC-01a v3): 값은 정확히 'True' / 'False' 문자열만 허용한다.
# 공백 제거·대소문자 변환을 하지 않는다.
GRAY_COLUMN = "gray"
GRAY_ALLOWED = {"True", "False"}

# 흑백 실제 사진이 있으면 안 되는 split (데이터 계약: train/val 흑백 실제는 컬러 reserve로 교체)
NO_GRAY_REAL_SPLITS = {"train", "val"}

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


def manifest_sha256(path) -> str:
    """manifest 파일 바이트 전체의 sha256 값(16진수 문자열)을 반환한다.

    출력된 해시를 data/manifests/manifest_vN.csv 스냅샷과 대조해
    실험에 쓴 manifest 버전을 확인하는 데 쓴다.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gray_problems(df: pd.DataFrame) -> list[str]:
    """gray 열 값 검사 (SPEC-01a v3). 빈 값을 먼저 걸러 따로 보고하고, 나머지에서 허용값을 본다."""
    values = df[GRAY_COLUMN]
    # 1. 빈 값('' 또는 공백만): 문자열 비교보다 먼저 본다 (빈 값이 조용히 False가 되지 않게)
    blank = values.str.strip() == ""
    # 2. 빈 값이 아닌데 정확히 'True'/'False'가 아님
    bad = ~blank & ~values.isin(GRAY_ALLOWED)

    problems = []
    if blank.any():
        examples = ", ".join(f"{_csv_line(i)}번 줄" for i in values[blank].head(MAX_EXAMPLES).index)
        problems.append(f"- 열 '{GRAY_COLUMN}'에 빈 값 {int(blank.sum())}개. 예: {examples}")
    if bad.any():
        examples = ", ".join(
            f"{_csv_line(i)}번 줄='{v}'" for i, v in values[bad].head(MAX_EXAMPLES).items()
        )
        problems.append(
            f"- 열 '{GRAY_COLUMN}'에 허용되지 않은 값 {int(bad.sum())}개 "
            f"(허용값: {sorted(GRAY_ALLOWED)}, 공백·대소문자 변환 없음). 예: {examples}"
        )
    return problems


def load_manifest(path, data_root) -> pd.DataFrame:
    """manifest CSV를 읽고 필수 열과 허용값(label, split, source, color_mode)을 확인한다.

    읽을 때 manifest 경로, 행 수, 파일의 sha256 해시를 출력한다.
    모든 값은 문자열로 읽는다 (origin_id 등이 숫자로 바뀌지 않게).
    data_root는 df.attrs["data_root"]에, 해시는 df.attrs["manifest_sha256"]에 기록해 둔다.

    gray는 선택 열이다 (v3). 없으면 모든 행을 False로 채우고, 있으면 값이 정확히
    'True'/'False'인지 검사한 뒤 bool로 바꾼다. df.attrs["gray_column"]에 "present"/"absent"를 기록한다.
    """
    # 바이트를 한 번만 읽어, 해시를 계산한 바로 그 내용을 파싱한다
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    print(f"manifest 읽기: {path} (행 {len(df)}개, sha256={digest})")

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
    gray_present = GRAY_COLUMN in df.columns
    if gray_present:
        problems.extend(_gray_problems(df))
    if problems:
        raise ManifestValidationError(
            f"manifest 허용값 검사 실패 (파일: {path}, 줄 번호는 헤더=1 기준)\n"
            + "\n".join(problems)
        )

    if gray_present:
        df[GRAY_COLUMN] = df[GRAY_COLUMN] == "True"
    else:
        # 파일은 건드리지 않는다 (sha256 그대로). 메모리의 df에만 열을 채운다
        df[GRAY_COLUMN] = False
        print(f"gray 열 없음 → 전부 False (행 {len(df)}개)")
    df[GRAY_COLUMN] = df[GRAY_COLUMN].astype(bool)
    df.attrs["gray_column"] = "present" if gray_present else "absent"

    df.attrs["data_root"] = str(data_root)
    df.attrs["manifest_sha256"] = digest
    return df


def _file_exists(full_path: Path) -> bool:
    return full_path.is_file()


def validate(df: pd.DataFrame, data_root, check_files="all") -> None:
    """데이터 계약 규칙 1~5를 모두 검사하고, 위반을 모아 하나의 에러로 보고한다.

    1. 한 origin_id가 두 개 이상의 split에 존재 (누수)
    2. 한 origin_id 안에서 label이 다름
    3. source가 kakao 또는 latest인데 split이 test가 아님
    4. path가 가리키는 파일이 없음 (check_files="all" 전체, 정수 n이면 n개 샘플)
    5. label == real, gray == True인데 split이 train 또는 val (v3)

    1번 위반이 있으면 LeakageError, 없으면 ManifestValidationError를 낸다.
    df는 load_manifest가 돌려준 것이어야 한다 (gray 열이 bool로 들어 있음).
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

    # 5. 흑백 실제 사진이 train/val에 남아 있음 (v3. 흑백 가짜는 위반 아님)
    gray = df[GRAY_COLUMN]
    is_real = df["label"] == "real"
    bad_gray = df[is_real & gray & df["split"].isin(NO_GRAY_REAL_SPLITS)]
    if len(bad_gray):
        examples = ", ".join(
            f"{_csv_line(i)}번 줄(split={r.split}, {r.path})"
            for i, r in bad_gray.head(MAX_EXAMPLES).iterrows()
        )
        violations.append(
            f"[5] 흑백 실제(label=real, gray=True)인데 split이 train/val인 행 {len(bad_gray)}개. 예: {examples}"
        )
    log.append(f"train/val 흑백 실제 {len(bad_gray)}개")
    n_gray_real_test = int((is_real & gray & (df["split"] == "test")).sum())
    fake_gray = df[(df["label"] == "fake") & gray]
    fake_counts = " / ".join(
        f"{s} {int((fake_gray['split'] == s).sum())}장" for s in ("train", "val", "test")
    )
    log.append(f"흑백 행: 실제 test {n_gray_real_test}장, 가짜 {fake_counts}")

    if violations:
        error_cls = LeakageError if overlap_ids else ManifestValidationError
        raise error_cls(
            f"manifest 검사 실패: 위반 {len(violations)}종류\n" + "\n".join(violations)
        )

    print(f"manifest 검사 통과 (행 {len(df)}개, origin_id {df['origin_id'].nunique()}개)")
    for line in log:
        print(f"  - {line}")


def gray_count_table(df: pd.DataFrame) -> pd.DataFrame:
    """label × split × gray 장수표 (v3). 행: (label, split), 열: False, True."""
    table = pd.crosstab([df["label"], df["split"]], df[GRAY_COLUMN])
    return table.reindex(columns=[False, True], fill_value=0)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """label × condition × split 장수표를 출력하고 반환한다.

    그 다음 label × split × gray 장수표를 출력한다 (v3, 반환값은 바꾸지 않음).
    original 조건에서 label별 format·크기 분포를 출력하고,
    label 사이에 분포가 다르면 경고만 한다 (멈추지 않는다).
    """
    table = pd.crosstab(
        [df["label"], df["condition"]], df["split"], dropna=False
    )
    print("[장수표] label × condition × split")
    print(table.to_string())

    print("[장수표] label × split × gray")
    print(gray_count_table(df).to_string())

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
