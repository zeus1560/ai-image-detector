"""평가 설정 읽기와 평가 전 확인 (SPEC-03 입력 1, 처리 방식 1-2, 처리 방식 5).

torch를 import하지 않는다 (로컬 pytest로 확인하기 위해 evaluate.py와 분리).

- load_eval_config: 필수 키, split·checkpoint 값 검사
- create_output_dir: run_dir/eval/{eval_name}/ 생성 (이미 있으면 에러)
- check_manifest_version: 학습 뒤 manifest가 바뀌었을 때 origin_id 누수 확인
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from .config import ConfigError
from .manifest import MAX_EXAMPLES, LeakageError

REQUIRED_KEYS = [
    "eval_name",
    "run_dir",
    "checkpoint",  # "best" 또는 "last"
    "manifest_path",
    "data_root",
    "split",  # "val" 또는 "test"
    "source",  # 문자열 또는 null (null이면 split의 모든 source)
    "batch_size",
    "num_workers",
]

# 평가 manifest 해시가 학습 때와 다를 때만 필요 (처리 방식 5)
OPTIONAL_KEYS = ["train_manifest_path"]

ALLOWED_SPLITS = ("val", "test")
ALLOWED_CHECKPOINTS = ("best", "last")


class OutputExistsError(FileExistsError):
    """평가 출력 폴더가 이미 있을 때 나는 에러 (이전 결과를 덮어쓰지 않기 위함)."""


class ManifestVersionError(ValueError):
    """manifest 버전 확인에 필요한 학습 manifest가 없거나 해시가 맞지 않을 때 나는 에러."""


def load_eval_config(path) -> dict:
    """평가 설정 JSON을 읽고 검사한다.

    - 필수 키가 빠지면 빠진 이름을 모두 담아 ConfigError
    - split은 "val" 또는 "test"만 허용 ("train"이면 에러)
    - checkpoint는 "best" 또는 "last"만 허용
    - split이 "test"면 checkpoint는 "best"만 허용
      (test에서 두 체크포인트를 보고 고르면 test로 모델을 선택하는 셈이기 때문)
    - train_manifest_path가 없으면 None으로 채운다
    """
    with open(Path(path), encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"평가 설정 파일의 최상위는 JSON 객체여야 합니다 (파일: {path})")

    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ConfigError(f"평가 설정 파일에 필수 키가 없습니다: {', '.join(missing)} (파일: {path})")

    if config["split"] not in ALLOWED_SPLITS:
        raise ConfigError(
            f"split은 {' 또는 '.join(ALLOWED_SPLITS)}만 허용합니다: {config['split']!r} (파일: {path})"
        )
    if config["checkpoint"] not in ALLOWED_CHECKPOINTS:
        raise ConfigError(
            f"checkpoint는 {' 또는 '.join(ALLOWED_CHECKPOINTS)}만 허용합니다: "
            f"{config['checkpoint']!r} (파일: {path})"
        )
    if config["split"] == "test" and config["checkpoint"] != "best":
        raise ConfigError(
            f"split=test 평가는 checkpoint=best만 허용합니다: {config['checkpoint']!r} (파일: {path}). "
            "test에서 체크포인트를 고르면 test로 모델을 선택하는 셈입니다."
        )

    for k in OPTIONAL_KEYS:
        config.setdefault(k, None)
    return config


def eval_output_dir(config: dict) -> Path:
    """평가 결과 폴더: run_dir/eval/{eval_name}/"""
    return Path(config["run_dir"]) / "eval" / config["eval_name"]


def _exists_error(path: Path) -> OutputExistsError:
    return OutputExistsError(
        f"평가 출력 폴더가 이미 있습니다: {path}. "
        "이전 평가 결과를 덮어쓰지 않으려면 eval_name을 바꾸세요."
    )


def ensure_output_dir_absent(path) -> None:
    """평가 결과 폴더가 이미 있으면 OutputExistsError (처리 방식 1-2, 만들지는 않는다).

    evaluate.py는 실행 초반에 이것으로 먼저 확인하고, manifest 검사·버전 확인이 통과한 뒤
    create_output_dir로 만든다 (검사 실패로 빈 폴더가 남아 같은 eval_name을 못 쓰게 되는 것을 막기 위함).
    """
    path = Path(path)
    if path.exists():
        raise _exists_error(path)


def create_output_dir(path) -> Path:
    """평가 결과 폴더를 만든다. 이미 있으면 OutputExistsError (처리 방식 1-2).

    폴더 안의 파일은 건드리지 않는다.
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise _exists_error(path) from None
    return path


def _read_train_manifest(path) -> tuple[pd.DataFrame, str]:
    """학습 manifest 스냅샷을 읽는다. 해시를 계산한 바로 그 바이트를 파싱한다."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    for col in ("origin_id", "split"):
        if col not in df.columns:
            raise ManifestVersionError(f"학습 manifest에 '{col}' 열이 없습니다: {path}")
    return df, digest


def check_manifest_version(
    eval_df: pd.DataFrame,
    split: str,
    eval_sha256: str,
    train_sha256: str,
    train_manifest_path=None,
) -> str:
    """학습 뒤 manifest가 바뀌었을 때 평가 대상 origin_id의 누수를 확인한다 (처리 방식 5).

    Args:
        eval_df: 평가 대상 행 (split·source로 거른 뒤). origin_id 열이 있어야 한다
        split: 평가 split ("val" 또는 "test")
        eval_sha256: 평가에 쓰는 manifest의 해시
        train_sha256: 학습 때 manifest의 해시 (run_dir/config.json의 manifest_sha256)
        train_manifest_path: 학습 때 manifest 스냅샷 파일 (해시가 다를 때만 필요)

    규칙
    - 해시가 같으면 통과
    - 다르면 train_manifest_path가 있어야 하고 그 파일 해시가 train_sha256과 같아야 한다 (아니면 ManifestVersionError)
    - split=val: 평가 대상 origin_id가 학습 manifest의 train origin_id와 겹치면 LeakageError
    - split=test: 학습 manifest의 train 또는 val origin_id와 겹치면 LeakageError
      (val은 best epoch 선택에 쓰였기 때문)
    - 학습 manifest에 없던 새 origin_id는 통과

    Returns:
        확인 결과 설명 (로그용 한 줄)
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split은 {ALLOWED_SPLITS} 중 하나여야 합니다: {split!r}")

    if eval_sha256 == train_sha256:
        return "manifest 버전 확인: 학습 때와 같은 manifest (해시 같음)"

    if train_manifest_path is None:
        raise ManifestVersionError(
            "평가 manifest 해시가 학습 때와 다릅니다. 학습 때 manifest 스냅샷을 "
            "설정 키 train_manifest_path로 지정하세요.\n"
            f"  평가 manifest_sha256: {eval_sha256}\n  학습 manifest_sha256: {train_sha256}"
        )
    train_df, digest = _read_train_manifest(train_manifest_path)
    if digest != train_sha256:
        raise ManifestVersionError(
            f"train_manifest_path 파일의 해시가 학습 때 해시와 다릅니다: {train_manifest_path}\n"
            f"  파일 sha256: {digest}\n  학습 manifest_sha256: {train_sha256}"
        )

    forbidden_splits = ("train",) if split == "val" else ("train", "val")
    forbidden = train_df[train_df["split"].isin(forbidden_splits)]
    forbidden_ids = set(forbidden["origin_id"])
    overlap = sorted(set(eval_df["origin_id"]) & forbidden_ids)
    if overlap:
        id_to_split = forbidden.drop_duplicates("origin_id").set_index("origin_id")["split"]
        examples = ", ".join(f"{oid}(학습 {id_to_split[oid]})" for oid in overlap[:MAX_EXAMPLES])
        raise LeakageError(
            f"split={split} 평가 대상 origin_id {len(overlap)}개가 학습 manifest의 "
            f"{'/'.join(forbidden_splits)}에 있었습니다 (데이터 누수). 예: {examples}"
        )

    n_new = len(set(eval_df["origin_id"]) - set(train_df["origin_id"]))
    return (
        "manifest 버전 확인: 해시 다름, 학습 manifest와 대조 통과 "
        f"(금지 split {'/'.join(forbidden_splits)}과 겹침 0개, 학습 manifest에 없던 origin_id {n_new}개)"
    )
