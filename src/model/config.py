"""학습 설정 파일 읽기와 필수 키 검사 (SPEC-02 입력 2).

torch를 import하지 않는다 (로컬 테스트를 위해 train.py와 분리).
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = [
    "run_name",
    "manifest_path",
    "data_root",
    "source",
    "crop_size",  # 정수 또는 null (null이면 크롭하지 않음, CIFAKE)
    "model",
    "pretrained",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
    "num_workers",
    "output_dir",
]


# 선택 키 (SPEC-02 확정 후 보완 4). 키가 없으면 아무 값도 채우지 않는다
# (채우면 기존 config.json과 check_resume_config 비교가 달라진다)
TRAIN_GENERATORS_KEY = "train_generators"


class ConfigError(ValueError):
    """설정 파일이 형식에 맞지 않을 때 나는 에러."""


def check_train_generators(value) -> None:
    """train_generators 형식 검사. 어기면 이유를 담아 ConfigError.

    생성기 이름 문자열의 목록이어야 한다: 빈 목록, 문자열이 아닌 값, 빈 문자열, 중복, "real" 포함은 에러.
    """
    key = TRAIN_GENERATORS_KEY
    if not isinstance(value, list):
        raise ConfigError(f"{key}는 생성기 이름 문자열의 목록이어야 합니다: {value!r}")
    if len(value) == 0:
        raise ConfigError(f"{key}가 빈 목록입니다 (키를 빼면 모든 생성기를 씁니다)")
    non_str = [v for v in value if not isinstance(v, str)]
    if non_str:
        raise ConfigError(f"{key}에 문자열이 아닌 값이 있습니다: {non_str!r}")
    if "" in value:
        raise ConfigError(f"{key}에 빈 문자열이 있습니다: {value!r}")
    dups = sorted({v for v in value if value.count(v) > 1})
    if dups:
        raise ConfigError(f"{key}에 중복된 생성기가 있습니다: {dups}")
    if "real" in value:
        raise ConfigError(f"{key}에 'real'이 있습니다 (실제 사진은 항상 전부 사용하므로 넣지 않습니다)")


def load_config(path) -> dict:
    """JSON 설정 파일을 읽고 필수 키가 모두 있는지 확인한다.

    빠진 키가 있으면 그 이름을 모두 담아 ConfigError를 낸다.
    값이 null인 키는 빠진 것으로 보지 않는다 (crop_size는 null이 허용된다).
    선택 키 train_generators가 있으면 형식을 검사한다 (보완 4). 없으면 그대로 둔다.
    """
    with open(Path(path), encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"설정 파일의 최상위는 JSON 객체여야 합니다 (파일: {path})")

    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ConfigError(f"설정 파일에 필수 키가 없습니다: {', '.join(missing)} (파일: {path})")
    if TRAIN_GENERATORS_KEY in config:
        check_train_generators(config[TRAIN_GENERATORS_KEY])
    return config
