"""학습 생성기 행 고르기 (SPEC-02 확정 후 보완 4).

torch를 import하지 않는다 (로컬 pytest로 확인). train.py는 ManifestDataset을 만든 뒤
dataset.df를 select_train_rows의 결과로 바꾼다. 학습 시작 로그 문자열도 여기서 만든다.
"""

from __future__ import annotations

import warnings

import pandas as pd

REAL_GENERATOR = "real"


class SelectionError(ValueError):
    """학습 행 고르기 검사 위반."""


def select_train_rows(df: pd.DataFrame, train_generators, split: str) -> pd.DataFrame:
    """학습 생성기에 맞는 행만 고른다 (원래 행 인덱스 유지).

    - train_generators가 None이면 df를 그대로 돌려준다.
    - 실제(label == real): 전부 사용 (공유 실제 세트)
    - 가짜(label == fake): generator가 train_generators에 있는 행만
    검사 (어기면 SelectionError, 생성기 이름과 split 출력):
      a. train_generators의 각 생성기가 이 split에서 가짜 행 0개
      b. 실제 행의 generator가 "real"이 아님
      c. 고른 행에서 origin_id 또는 path가 두 번 이상 나옴
    고른 결과 real:fake 장수가 다르면 경고만 한다.
    """
    if train_generators is None:
        return df

    is_real = df["label"] == "real"
    is_fake = df["label"] == "fake"
    problems = []

    # a. 지정한 생성기의 가짜 행이 이 split에 없음 (오타로 가짜 없이 학습하는 것을 막음)
    fake_gens = df.loc[is_fake, "generator"]
    absent = [g for g in train_generators if not (fake_gens == g).any()]
    if absent:
        problems.append(f"[a] split={split}에 가짜 행이 0개인 학습 생성기: {absent}")

    # b. 실제 행의 generator는 real이어야 함 (데이터 계약)
    bad_real = sorted(set(df.loc[is_real & (df["generator"] != REAL_GENERATOR), "generator"]))
    if bad_real:
        n = int((is_real & (df["generator"] != REAL_GENERATOR)).sum())
        problems.append(f"[b] split={split}에 generator가 real이 아닌 실제 행 {n}개. generator 값: {bad_real}")

    selected = df[is_real | (is_fake & df["generator"].isin(train_generators))]

    # c. 고른 행 안의 origin_id·path 중복 (original은 원본당 1행)
    for col in ("origin_id", "path"):
        dup = selected[selected[col].duplicated(keep=False)]
        if len(dup):
            values = list(dict.fromkeys(dup[col]))
            gens = sorted(set(dup["generator"]))
            problems.append(
                f"[c] split={split}에서 {col}가 두 번 이상 나오는 값 {len(values)}개 "
                f"(관련 generator: {gens}). 예: {values[:10]}"
            )

    if problems:
        raise SelectionError(
            f"학습 생성기 행 고르기 실패 (train_generators={list(train_generators)})\n" + "\n".join(problems)
        )

    n_real = int((selected["label"] == "real").sum())
    n_fake = int((selected["label"] == "fake").sum())
    if n_real != n_fake:
        warnings.warn(
            f"split={split} 고른 결과 real {n_real} : fake {n_fake} 장수가 다릅니다 (보정하지 않고 계속)",
            UserWarning,
            stacklevel=2,
        )
    return selected


# ---------------------------------------------------------------- 학습 시작 로그 (보완 4의 3)


def train_generators_line(train_generators) -> str:
    if train_generators is None:
        return "train_generators: 키 없음: 모든 생성기 사용"
    return f"train_generators: {list(train_generators)}"


def generator_table_lines(before: dict[str, pd.DataFrame], train_generators) -> list[str]:
    """고르기 전 split × generator × label 행 수표. 각 줄에 사용/제외 표시.

    before: split → 고르기 전 df (source·original·split으로 거른 ManifestDataset.df)
    """
    lines = ["[고르기 전] split × generator × label 행 수 (source·original 기준)"]
    for split, df in before.items():
        counts = df.groupby(["generator", "label"]).size()
        for (gen, label), n in counts.items():
            used = (
                train_generators is None
                or label == "real"
                or (label == "fake" and gen in train_generators)
            )
            lines.append(f"  {split:5s} {gen:12s} {label:4s} {int(n):>8d}  {'사용' if used else '제외'}")
    return lines


def gray_table_lines(after: dict[str, pd.DataFrame]) -> list[str]:
    """고른 뒤 split × label × gray 행 수 (흑백 가짜가 train/val에 얼마나 남았는지 확인용)."""
    lines = ["[고른 뒤] split × label × gray 행 수"]
    for split, df in after.items():
        for label in ("real", "fake"):
            sel = df["label"] == label
            n_true = int((sel & df["gray"]).sum())
            n_false = int((sel & ~df["gray"]).sum())
            lines.append(f"  {split:5s} {label:4s} gray=False {n_false:>8d}, gray=True {n_true:>8d}")
    return lines
