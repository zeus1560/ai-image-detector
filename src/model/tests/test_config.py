"""SPEC-02 A: config.py 테스트."""

import json

import pytest

from src.model.config import REQUIRED_KEYS, ConfigError, load_config


def full_config():
    return {
        "run_name": "cifake_baseline",
        "manifest_path": "/content/data/cifake_manifest.csv",
        "data_root": "/content/data",
        "source": "cifake",
        "crop_size": None,
        "model": "resnet50",
        "pretrained": True,
        "epochs": 5,
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 0,
        "seed": 42,
        "num_workers": 2,
        "output_dir": "/content/drive/MyDrive/ai-image-detector-runs",
    }


def write(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_full_config_loads_with_null_crop(tmp_path):
    config = load_config(write(tmp_path, full_config()))
    assert set(REQUIRED_KEYS) <= config.keys()
    assert config["crop_size"] is None  # null은 빠진 키가 아니다


def test_missing_keys_listed_in_error(tmp_path):
    """A10: 필수 키가 빠지면 에러, 빠진 키 이름이 모두 출력된다."""
    data = full_config()
    del data["seed"]
    del data["crop_size"]
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, data))
    assert "seed" in str(exc.value)
    assert "crop_size" in str(exc.value)


def test_non_object_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, [1, 2, 3]))
