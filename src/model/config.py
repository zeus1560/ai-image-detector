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


class ConfigError(ValueError):
    """설정 파일이 형식에 맞지 않을 때 나는 에러."""


def load_config(path) -> dict:
    """JSON 설정 파일을 읽고 필수 키가 모두 있는지 확인한다.

    빠진 키가 있으면 그 이름을 모두 담아 ConfigError를 낸다.
    값이 null인 키는 빠진 것으로 보지 않는다 (crop_size는 null이 허용된다).
    """
    with open(Path(path), encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"설정 파일의 최상위는 JSON 객체여야 합니다 (파일: {path})")

    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ConfigError(f"설정 파일에 필수 키가 없습니다: {', '.join(missing)} (파일: {path})")
    return config
