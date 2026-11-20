"""SPEC-03 A: eval_config.py 테스트 (설정 검사, 출력 폴더, manifest 버전 확인)."""

import hashlib
import json

import pandas as pd
import pytest

from src.model.config import ConfigError
from src.model.eval_config import (
    REQUIRED_KEYS,
    ManifestVersionError,
    OutputExistsError,
    check_manifest_version,
    create_output_dir,
    eval_output_dir,
    load_eval_config,
)
from src.model.manifest import LeakageError


def full_config(**overrides):
    config = {
        "eval_name": "cifake_baseline_eval_val",
        "run_dir": "/content/drive/MyDrive/ai-image-detector-runs/cifake_baseline",
        "checkpoint": "best",
        "manifest_path": "/content/data/cifake_manifest.csv",
        "data_root": "/content/data",
        "split": "val",
        "source": "cifake",
        "batch_size": 256,
        "num_workers": 2,
    }
    config.update(overrides)
    return config


def write(tmp_path, data):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 설정 검사


def test_full_config_loads(tmp_path):
    config = load_eval_config(write(tmp_path, full_config(source=None)))
    assert set(REQUIRED_KEYS) <= config.keys()
    assert config["source"] is None  # null은 빠진 키가 아니다
    assert config["train_manifest_path"] is None  # 선택 키는 None으로 채운다


def test_missing_keys_listed_in_error(tmp_path):
    """A1: 필수 키가 빠지면 에러, 빠진 키 이름이 출력된다."""
    data = full_config()
    del data["split"]
    del data["run_dir"]
    with pytest.raises(ConfigError) as exc:
        load_eval_config(write(tmp_path, data))
    assert "split" in str(exc.value)
    assert "run_dir" in str(exc.value)


def test_split_train_rejected(tmp_path):
    """A1: split이 train이면 에러."""
    with pytest.raises(ConfigError, match="split"):
        load_eval_config(write(tmp_path, full_config(split="train")))


def test_unknown_checkpoint_rejected(tmp_path):
    with pytest.raises(ConfigError, match="checkpoint"):
        load_eval_config(write(tmp_path, full_config(checkpoint="epoch3")))


def test_test_split_with_last_rejected(tmp_path):
    """A9: split=test + checkpoint=last면 에러."""
    with pytest.raises(ConfigError, match="test"):
        load_eval_config(write(tmp_path, full_config(split="test", checkpoint="last")))


def test_val_split_allows_best_and_last(tmp_path):
    for ckpt in ("best", "last"):
        assert load_eval_config(write(tmp_path, full_config(checkpoint=ckpt)))["checkpoint"] == ckpt
    assert load_eval_config(write(tmp_path, full_config(split="test")))["checkpoint"] == "best"


# ---------------------------------------------------------------- 출력 폴더


def test_output_dir_exists_raises_and_keeps_files(tmp_path):
    """A8: 출력 폴더가 이미 있으면 에러가 나고, 기존 파일은 그대로다."""
    config = full_config(run_dir=str(tmp_path / "run"))
    out = create_output_dir(eval_output_dir(config))
    assert out == tmp_path / "run" / "eval" / "cifake_baseline_eval_val"
    (out / "predictions.csv").write_text("old", encoding="utf-8")

    with pytest.raises(OutputExistsError):
        create_output_dir(eval_output_dir(config))
    assert (out / "predictions.csv").read_text(encoding="utf-8") == "old"


# ---------------------------------------------------------------- manifest 버전 확인 (A7)

# 학습 manifest: t* = train, v* = val, s* = test
TRAIN_ROWS = [("t1", "train"), ("t2", "train"), ("v1", "val"), ("v2", "val"), ("s1", "test")]


@pytest.fixture
def train_manifest(tmp_path):
    """학습 때 manifest 스냅샷 파일과 그 해시."""
    df = pd.DataFrame(TRAIN_ROWS, columns=["origin_id", "split"])
    path = tmp_path / "manifest_v1.csv"
    df.to_csv(path, index=False)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def eval_rows(*origin_ids):
    return pd.DataFrame({"origin_id": list(origin_ids)})


def test_version_same_hash_passes():
    """A7: 해시 같음 → 통과 (학습 manifest 파일 불필요)."""
    msg = check_manifest_version(eval_rows("t1"), "val", "aaa", "aaa", None)
    assert "해시 같음" in msg


def test_version_val_eval_overlapping_only_train_val_passes(train_manifest):
    """A7: 해시 다름 + val 평가 + 학습 val과만 겹침 → 통과. 새 origin_id도 통과."""
    path, sha = train_manifest
    msg = check_manifest_version(eval_rows("v1", "v2", "new1"), "val", "new-hash", sha, path)
    assert "통과" in msg


def test_version_val_eval_overlapping_train_raises(train_manifest):
    """A7: 해시 다름 + val 평가 + 학습 train과 겹침 → 에러."""
    path, sha = train_manifest
    with pytest.raises(LeakageError, match="t1"):
        check_manifest_version(eval_rows("v1", "t1"), "val", "new-hash", sha, path)


def test_version_test_eval_overlapping_train_val_raises(train_manifest):
    """A7: 해시 다름 + test 평가 + 학습 val과 겹침 → 에러."""
    path, sha = train_manifest
    with pytest.raises(LeakageError, match="v1"):
        check_manifest_version(eval_rows("s1", "v1"), "test", "new-hash", sha, path)


def test_version_test_eval_new_and_test_ids_pass(train_manifest):
    path, sha = train_manifest
    check_manifest_version(eval_rows("s1", "new1"), "test", "new-hash", sha, path)


def test_version_train_manifest_hash_mismatch_raises(train_manifest):
    """A7: 학습 manifest 파일 해시가 학습 때 해시와 다름 → 에러."""
    path, _ = train_manifest
    with pytest.raises(ManifestVersionError, match="해시"):
        check_manifest_version(eval_rows("s1"), "test", "new-hash", "not-the-train-hash", path)


def test_version_missing_train_manifest_path_raises():
    with pytest.raises(ManifestVersionError, match="train_manifest_path"):
        check_manifest_version(eval_rows("s1"), "test", "new-hash", "old-hash", None)
