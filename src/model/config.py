"""학습 설정 파일 읽기와 필수 키 검사 (SPEC-02 입력 2), augment 검사와 공정 비교 (SPEC-04).

torch를 import하지 않는다 (로컬 테스트를 위해 train.py와 분리).
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path

from .resume import ENV_KEYS

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

# 선택 키 (SPEC-04 입력 2). 키가 없으면 증강 없음이고 기본값을 채우지 않는다
AUGMENT_KEY = "augment"
AUGMENT_FIELDS = ("jpeg_prob", "jpeg_quality_min", "jpeg_quality_max")
QUALITY_RANGE = (1, 100)

# check_same_setup에서 달라도 되는 키 (SPEC-04 처리 방식 5, 보완 1-3)
SAME_SETUP_FREE_KEYS = ("run_name", AUGMENT_KEY, "num_workers")


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


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_augment(value) -> None:
    """augment 형식 검사 (SPEC-04 입력 2). 어기면 이유를 담아 ConfigError.

    jpeg_prob: 0 이상 1 이하 숫자. jpeg_quality_min·max: 1~100 정수, min ≤ max.
    세 키 외의 키가 있거나 하나라도 빠지면 에러.
    """
    key = AUGMENT_KEY
    if not isinstance(value, dict):
        raise ConfigError(f"{key}는 JSON 객체여야 합니다: {value!r}")
    missing = [k for k in AUGMENT_FIELDS if k not in value]
    if missing:
        raise ConfigError(f"{key}에 필요한 키가 없습니다: {missing}")
    unknown = [k for k in value if k not in AUGMENT_FIELDS]
    if unknown:
        raise ConfigError(f"{key}에 모르는 키가 있습니다: {unknown} (허용: {list(AUGMENT_FIELDS)})")

    prob = value["jpeg_prob"]
    if not isinstance(prob, numbers.Real) or isinstance(prob, bool):
        raise ConfigError(f"{key}.jpeg_prob는 숫자여야 합니다: {prob!r}")
    if not 0 <= prob <= 1:
        raise ConfigError(f"{key}.jpeg_prob는 0 이상 1 이하여야 합니다: {prob!r}")

    lo, hi = QUALITY_RANGE
    for name in ("jpeg_quality_min", "jpeg_quality_max"):
        q = value[name]
        if not _is_int(q):
            raise ConfigError(f"{key}.{name}는 정수여야 합니다: {q!r}")
        if not lo <= q <= hi:
            raise ConfigError(f"{key}.{name}는 {lo}~{hi} 범위여야 합니다: {q!r}")
    if value["jpeg_quality_min"] > value["jpeg_quality_max"]:
        raise ConfigError(
            f"{key}.jpeg_quality_min이 jpeg_quality_max보다 큽니다: "
            f"{value['jpeg_quality_min']} > {value['jpeg_quality_max']}"
        )


_MISSING = object()


def check_same_setup(baseline_record: dict, aug_record: dict) -> list[str]:
    """베이스라인과 증강 run의 config.json 내용을 비교한다 (SPEC-04 처리 방식 5, 보완 1-1·1-3).

    - SAME_SETUP_FREE_KEYS(run_name, augment, num_workers)는 달라도 된다.
    - ENV_KEYS(git 커밋, torch·torchvision·Pillow·libjpeg 버전, GPU 이름)가 다르면 경고 문자열 목록을 돌려준다.
    - 그 밖의 키가 하나라도 다르면(한쪽에만 있는 키 포함) ConfigError, 다른 키 이름을 출력한다.
    """
    skip = set(SAME_SETUP_FREE_KEYS) | set(ENV_KEYS)
    keys = [k for k in dict.fromkeys([*baseline_record, *aug_record]) if k not in skip]
    differ = [k for k in keys if baseline_record.get(k, _MISSING) != aug_record.get(k, _MISSING)]
    if differ:
        raise ConfigError(f"베이스라인과 증강 run의 설정이 다릅니다 (공정 비교 불가). 다른 키: {differ}")
    warnings = []
    for k in ENV_KEYS:
        old, new = baseline_record.get(k), aug_record.get(k)
        if old != new:
            warnings.append(f"{k}가 베이스라인과 다릅니다: {old} → {new}")
    return warnings


def load_config(path) -> dict:
    """JSON 설정 파일을 읽고 필수 키가 모두 있는지 확인한다.

    빠진 키가 있으면 그 이름을 모두 담아 ConfigError를 낸다.
    값이 null인 키는 빠진 것으로 보지 않는다 (crop_size는 null이 허용된다).
    선택 키 train_generators가 있으면 형식을 검사한다 (보완 4). 없으면 그대로 둔다.
    선택 키 augment도 같은 방식이다 (SPEC-04).
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
    if AUGMENT_KEY in config:
        check_augment(config[AUGMENT_KEY])
    return config
