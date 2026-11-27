"""SPEC-04 A: 압축 증강 (config.py augment 검사·check_same_setup, transforms.py jpeg_plan·jpeg_recompress·CropWrapper).

torch 없이 돈다.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.model import resume as rs
from src.model import transforms as tf
from src.model.config import (
    AUGMENT_KEY,
    ConfigError,
    check_same_setup,
    load_config,
)
from src.model.transforms import (
    CropWrapper,
    jpeg_plan,
    jpeg_plan_stats,
    jpeg_recompress,
    train_crop,
    train_crop_origin,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
AUG = {"jpeg_prob": 0.5, "jpeg_quality_min": 30, "jpeg_quality_max": 95}


def base_config():
    with open(CONFIGS_DIR / "cifake_baseline.json", encoding="utf-8") as f:
        return json.load(f)


def write(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def noise_image(width, height, seed=0):
    """JPEG 손실이 품질에 따라 잘 드러나도록 부드러운 그라데이션 + 잡음을 섞은 RGB 이미지."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.int16)
    arr = np.clip(base + rng.integers(-40, 41, size=base.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ================================================================ config: augment 키


def test_config_without_augment_unchanged_and_resume_compare_ok():
    """A1: augment 없는 설정은 지금과 같게 읽히고, 기존 config.json과 check_resume_config에서 에러 없이 비교된다.

    기존 config.json = SPEC-04 이전 train.py가 쓴 {설정, manifest_sha256, 실행 환경 4키}.
    이번 실행은 Pillow·libjpeg 두 키가 더 있으므로 경고 2줄이 늘어난다 (보완 1-1에서 허용).
    """
    config = load_config(CONFIGS_DIR / "cifake_baseline.json")
    assert AUGMENT_KEY not in config  # 기본값을 채우지 않는다
    assert config == base_config()

    sha = "79e176eeb61bb024e6ffbc5163b5c84fa02f23daf74b9e39c2427086bcb3740c"
    old_env = {"git_commit": "old", "torch_version": "2.x", "torchvision_version": "0.x", "gpu_name": "T4"}
    saved = {**base_config(), rs.MANIFEST_KEY: sha, **old_env}
    current = {
        **config,
        rs.MANIFEST_KEY: sha,
        **old_env,
        "pillow_version": "11.3.0",
        "libjpeg_version": "8.0",
    }
    warnings_ = rs.check_resume_config(saved, current)
    assert warnings_ == [
        "pillow_version가 처음 실행과 다릅니다: None → 11.3.0",
        "libjpeg_version가 처음 실행과 다릅니다: None → 8.0",
    ]

    # 대조: 키 없을 때 None을 채웠다면 재개가 막혔을 것이다
    with pytest.raises(rs.ResumeError):
        rs.check_resume_config(saved, {**current, AUGMENT_KEY: None})


def test_valid_augment_loaded(tmp_path):
    data = {**base_config(), AUGMENT_KEY: dict(AUG)}
    assert load_config(write(tmp_path, data))[AUGMENT_KEY] == AUG
    # 경계값도 허용: prob 0·1, 품질 1·100, min == max
    for aug in (
        {"jpeg_prob": 0, "jpeg_quality_min": 1, "jpeg_quality_max": 100},
        {"jpeg_prob": 1.0, "jpeg_quality_min": 50, "jpeg_quality_max": 50},
    ):
        assert load_config(write(tmp_path, {**base_config(), AUGMENT_KEY: aug}))[AUGMENT_KEY] == aug


def test_smoke_config_is_baseline_plus_augment():
    """cifake_aug_smoke.json = cifake_baseline + run_name·epochs·augment (나머지 키 동일)."""
    smoke = load_config(CONFIGS_DIR / "cifake_aug_smoke.json")
    base = base_config()
    assert smoke[AUGMENT_KEY] == AUG
    assert smoke["epochs"] == 1
    differ = sorted(k for k in {*smoke, *base} if smoke.get(k) != base.get(k))
    assert differ == ["augment", "epochs", "run_name"]


@pytest.mark.parametrize(
    "aug, reason",
    [
        ({**AUG, "jpeg_prob": -0.1}, "0 이상 1 이하"),
        ({**AUG, "jpeg_prob": 1.1}, "0 이상 1 이하"),
        ({**AUG, "jpeg_prob": "0.5"}, "숫자"),
        ({**AUG, "jpeg_prob": True}, "숫자"),
        ({**AUG, "jpeg_quality_min": 30.5}, "정수"),
        ({**AUG, "jpeg_quality_max": 95.0}, "정수"),
        ({**AUG, "jpeg_quality_min": "30"}, "정수"),
        ({**AUG, "jpeg_quality_max": True}, "정수"),
        ({**AUG, "jpeg_quality_min": 96, "jpeg_quality_max": 95}, "보다 큽니다"),
        ({**AUG, "jpeg_quality_min": 0}, "1~100"),
        ({**AUG, "jpeg_quality_max": 101}, "1~100"),
        ({"jpeg_prob": 0.5, "jpeg_quality_min": 30}, "필요한 키가 없습니다"),
        ({**AUG, "jpeg_quality": 50}, "모르는 키"),
        ([0.5, 30, 95], "JSON 객체"),
        (None, "JSON 객체"),
    ],
)
def test_bad_augment_raises_config_error_with_reason(tmp_path, aug, reason):
    """A2: 범위 밖·정수 아님·min>max·1~100 밖·키 누락·모르는 키가 각각 ConfigError이고 이유가 출력된다."""
    data = {**base_config(), AUGMENT_KEY: aug}
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, data))
    assert reason in str(exc.value)
    assert AUGMENT_KEY in str(exc.value)


# ================================================================ jpeg_plan


def test_jpeg_plan_deterministic_and_order_independent():
    """A3: 같은 (seed, epoch, row_index)면 같은 결과, 호출 순서를 섞어도 같다."""
    keys = [(seed, epoch, row) for seed in (0, 42) for epoch in (1, 2) for row in range(200)]
    first = {k: jpeg_plan(*k, AUG) for k in keys}
    second = {k: jpeg_plan(*k, AUG) for k in keys}
    assert first == second
    shuffled = keys[:]
    random.Random(3).shuffle(shuffled)
    assert {k: jpeg_plan(*k, AUG) for k in shuffled} == first


def test_jpeg_plan_distribution_10000_rows():
    """A4: row 10,000개에서 적용 비율 0.47~0.53, 품질 모두 30~95, 30과 95가 모두 나온다."""
    plans = [jpeg_plan(42, 1, r, AUG) for r in range(10_000)]
    qualities = [q for q in plans if q is not None]
    ratio = len(qualities) / len(plans)
    assert 0.47 <= ratio <= 0.53, ratio
    assert all(isinstance(q, int) and 30 <= q <= 95 for q in qualities)
    assert 30 in qualities and 95 in qualities


def test_jpeg_plan_prob_zero_and_one():
    """A5: jpeg_prob=0이면 모두 None, 1이면 모두 품질 값."""
    zero = {**AUG, "jpeg_prob": 0}
    one = {**AUG, "jpeg_prob": 1}
    assert all(jpeg_plan(42, 1, r, zero) is None for r in range(2000))
    assert all(isinstance(jpeg_plan(42, 1, r, one), int) for r in range(2000))


def test_jpeg_plan_changes_with_epoch():
    """A6: epoch만 바꾸면 결과가 달라지는 row가 있다."""
    e1 = [jpeg_plan(42, 1, r, AUG) for r in range(500)]
    e2 = [jpeg_plan(42, 2, r, AUG) for r in range(500)]
    assert e1 != e2


def test_jpeg_plan_none_augment_returns_none():
    assert all(jpeg_plan(42, 1, r, None) is None for r in range(50))


def test_jpeg_plan_uses_separate_stream_from_crop():
    """압축 난수원은 [seed, epoch, row_index, 1]이다 (크롭 난수원 [seed, epoch, row_index]와 다름)."""
    for r in range(50):
        rng = np.random.default_rng([42, 3, r, 1])
        u = rng.random()
        expected = int(rng.integers(30, 96)) if u < 0.5 else None
        assert jpeg_plan(42, 3, r, AUG) == expected


# ================================================================ jpeg_plan_stats (보완 1-2)


def test_jpeg_plan_stats_labels_do_not_change_plans():
    """보완 1-2: 같은 row_indices에서 labels만 바꿔도 각 행의 계획이 같다.

    label이 한 행씩인 집계로 각 행의 계획(적용 여부·품질)을 꺼내 비교한다.
    """
    rows = list(range(100, 400))
    plans = [jpeg_plan(7, 2, r, AUG) for r in rows]

    def per_row(labels):
        # 행마다 고유 label을 붙이면 label별 통계가 곧 그 행의 계획이다
        stats = jpeg_plan_stats(7, 2, rows, labels, AUG)
        return [stats[lab]["q_min"] if stats[lab]["applied"] else None for lab in labels]

    labels_a = [f"a{i:04d}" for i in range(len(rows))]
    labels_b = [f"b{i:04d}" for i in reversed(range(len(rows)))]
    assert per_row(labels_a) == plans
    assert per_row(labels_b) == plans

    # real/fake를 서로 바꿔 붙여도 전체 적용 수·품질 합계는 같다
    rf = ["real" if i % 2 else "fake" for i in range(len(rows))]
    fr = ["fake" if i % 2 else "real" for i in range(len(rows))]
    s1, s2 = jpeg_plan_stats(7, 2, rows, rf, AUG), jpeg_plan_stats(7, 2, rows, fr, AUG)
    total = lambda s: sum(v["applied"] for v in s.values())  # noqa: E731
    assert total(s1) == total(s2) == sum(q is not None for q in plans)
    assert s1["real"]["applied"] == s2["fake"]["applied"]


def test_jpeg_plan_stats_matches_hand_calculation(monkeypatch):
    """보완 1-2: 손으로 만든 예시에서 label별 통계가 손 계산 값과 같다.

    jpeg_plan을 고정 표로 바꿔 계획을 손으로 정한다.
    real: 행 0(→40), 1(None), 2(→60), 3(None) → 2/4, 평균 50, 최소 40, 최대 60
    fake: 행 4(→90), 5(→30), 6(→60)          → 3/3, 평균 60, 최소 30, 최대 90
    gray: 행 7(None)                          → 0/1, 품질 없음
    """
    table = {0: 40, 1: None, 2: 60, 3: None, 4: 90, 5: 30, 6: 60, 7: None}
    calls = []

    def fake_plan(seed, epoch, row_index, augment):
        calls.append((seed, epoch, row_index))
        return table[row_index]

    monkeypatch.setattr(tf, "jpeg_plan", fake_plan)
    labels = ["real"] * 4 + ["fake"] * 3 + ["gray"]
    stats = jpeg_plan_stats(42, 5, range(8), labels, AUG)

    assert list(stats) == ["fake", "gray", "real"]  # label 이름순
    assert stats["real"] == {"count": 4, "applied": 2, "ratio": 0.5, "q_mean": 50.0, "q_min": 40, "q_max": 60}
    assert stats["fake"] == {"count": 3, "applied": 3, "ratio": 1.0, "q_mean": 60.0, "q_min": 30, "q_max": 90}
    assert stats["gray"] == {"count": 1, "applied": 0, "ratio": 0.0, "q_mean": None, "q_min": None, "q_max": None}
    # label이 계획 함수에 넘어가지 않는다 (seed, epoch, row_index만)
    assert calls == [(42, 5, r) for r in range(8)]


def test_jpeg_plan_stats_length_mismatch_raises():
    with pytest.raises(ValueError):
        jpeg_plan_stats(42, 1, [1, 2, 3], ["real", "fake"], AUG)


# ================================================================ jpeg_recompress


def mean_abs_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).mean())


def test_jpeg_recompress_keeps_size_and_mode_and_quality_order():
    """A7: 512×512 RGB → 512×512 RGB, 품질 30 결과가 품질 95보다 원본과의 평균 절대 차이가 크다."""
    img = noise_image(512, 512)
    q30 = jpeg_recompress(img, 30)
    q95 = jpeg_recompress(img, 95)
    for out in (q30, q95):
        assert out.size == (512, 512)
        assert out.mode == "RGB"
    assert mean_abs_diff(img, q30) > mean_abs_diff(img, q95)


def test_jpeg_recompress_small_image_and_deterministic():
    """32×32(CIFAKE)도 크기 유지, 같은 입력·품질이면 같은 결과."""
    img = noise_image(32, 32, seed=1)
    a, b = jpeg_recompress(img, 50), jpeg_recompress(img, 50)
    assert a.size == (32, 32)
    assert a.tobytes() == b.tobytes()


# ================================================================ CropWrapper + augment


class FakeDataset:
    """(PIL, label, row_index)를 돌려주고 df.path를 갖는 가짜 dataset. row_index는 위치와 다르다."""

    def __init__(self, sizes, row_indices):
        self.df = pd.DataFrame({"path": [f"images/fake/{r}.png" for r in row_indices]}, index=row_indices)
        self.images = [noise_image(w, h, seed=r) for (w, h), r in zip(sizes, row_indices)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], i % 2, int(self.df.index[i])


def identity(img):
    return img


ROWS = tuple(range(1000, 1012))


def make_wrapper(mode="train", augment=None, crop_size=256, seed=42, size=(512, 512), rows=ROWS):
    ds = FakeDataset([size] * len(rows), list(rows))
    return CropWrapper(ds, crop_size, mode, seed, identity, augment=augment)


def test_wrapper_crop_coords_same_with_and_without_augment(monkeypatch):
    """A8: augment 있음/없음의 크롭 좌표가 같은 (seed, epoch, row_index)에서 같다."""
    recorded = []
    original = tf.train_crop_origin

    def spy(width, height, crop_size, seed, epoch, row_index, path=None):
        out = original(width, height, crop_size, seed, epoch, row_index, path)
        recorded.append(((seed, epoch, row_index), out))
        return out

    monkeypatch.setattr(tf, "train_crop_origin", spy)

    coords = {}
    for name, aug in (("none", None), ("aug", {**AUG, "jpeg_prob": 1})):
        recorded.clear()
        w = make_wrapper(augment=aug)
        for epoch in (1, 2):
            w.set_epoch(epoch)
            for i in range(len(w)):
                w[i]
        coords[name] = list(recorded)
    assert coords["none"] == coords["aug"]
    assert len(coords["none"]) == 2 * len(ROWS)
    # 좌표는 크롭 난수원 그대로다
    for (seed, epoch, row), xy in coords["aug"]:
        assert xy == original(512, 512, 256, seed, epoch, row)


def test_wrapper_compresses_full_image_before_crop():
    """A9: jpeg_prob=1이면 출력 = (512 전체 jpeg_recompress → 같은 좌표로 크롭), 픽셀 단위로 같다."""
    aug = {**AUG, "jpeg_prob": 1}
    w = make_wrapper(augment=aug)
    for epoch in (1, 2):
        w.set_epoch(epoch)
        for i in range(len(w)):
            out, label, row_index = w[i]
            q = jpeg_plan(42, epoch, row_index, aug)
            assert q is not None
            full = jpeg_recompress(w.dataset.images[i], q)
            expected = train_crop(full, 256, 42, epoch, row_index)
            assert out.size == (256, 256)
            assert out.tobytes() == expected.tobytes()
            # 크롭 후 압축과는 다르다 (순서가 결과를 바꾸는지 대조)
            x, y = train_crop_origin(512, 512, 256, 42, epoch, row_index)
            crop_then_jpeg = jpeg_recompress(w.dataset.images[i].crop((x, y, x + 256, y + 256)), q)
            if (x, y) != (0, 0):
                assert out.tobytes() != crop_then_jpeg.tobytes()


def test_wrapper_mixed_prob_compresses_only_planned_rows():
    """jpeg_prob=0.5: 계획이 None인 행은 원본 크롭 그대로, 품질이 있는 행은 압축 후 크롭."""
    w = make_wrapper(augment=AUG, rows=tuple(range(2000, 2040)))
    w.set_epoch(3)
    seen = {True: 0, False: 0}
    for i in range(len(w)):
        out, _, row_index = w[i]
        q = jpeg_plan(42, 3, row_index, AUG)
        src = w.dataset.images[i] if q is None else jpeg_recompress(w.dataset.images[i], q)
        assert out.tobytes() == train_crop(src, 256, 42, 3, row_index).tobytes()
        seen[q is not None] += 1
    assert seen[True] > 0 and seen[False] > 0


def test_wrapper_eval_with_augment_does_not_compress():
    """A10: CropWrapper(eval, augment 있음)는 압축하지 않는다 (augment 없는 eval과 픽셀 단위로 같음)."""
    with_aug = make_wrapper(mode="eval", augment={**AUG, "jpeg_prob": 1})
    without = make_wrapper(mode="eval", augment=None)
    for i in range(len(with_aug)):
        a, _, _ = with_aug[i]
        b, _, _ = without[i]
        assert a.tobytes() == b.tobytes()


def test_wrapper_null_crop_with_augment_32px():
    """CIFAKE(crop_size=None, 32×32)에서도 압축만 하고 크기는 그대로다."""
    aug = {**AUG, "jpeg_prob": 1}
    w = make_wrapper(augment=aug, crop_size=None, size=(32, 32))
    w.set_epoch(1)
    for i in range(len(w)):
        out, _, row_index = w[i]
        assert out.size == (32, 32)
        expected = jpeg_recompress(w.dataset.images[i], jpeg_plan(42, 1, row_index, aug))
        assert out.tobytes() == expected.tobytes()


def test_wrapper_augment_default_is_none():
    """augment 인자를 안 주면 기존과 같다 (압축 없음)."""
    ds = FakeDataset([(512, 512)] * 3, [5, 6, 7])
    w = CropWrapper(ds, 256, "train", 42, identity)
    assert w.augment is None
    w.set_epoch(1)
    for i in range(len(w)):
        out, _, row_index = w[i]
        assert out.tobytes() == train_crop(ds.images[i], 256, 42, 1, row_index).tobytes()


# ================================================================ check_same_setup


def record(**overrides):
    """config.json 형식: 설정 + manifest_sha256 + 실행 환경 6키."""
    rec = {
        **base_config(),
        "run_name": "genimage_baseline",
        "train_generators": ["sd14"],
        "manifest_sha256": "aaa",
        "git_commit": "c1",
        "torch_version": "2.5.0",
        "torchvision_version": "0.20.0",
        "gpu_name": "Tesla T4",
        "pillow_version": "11.3.0",
        "libjpeg_version": "8.0",
    }
    rec.update(overrides)
    return rec


def test_same_setup_only_free_keys_differ_passes():
    """A11-1: run_name·augment만 다르면 통과 (경고 없음)."""
    base = record()
    aug = record(run_name="genimage_aug", augment=dict(AUG))
    assert check_same_setup(base, aug) == []


def test_same_setup_num_workers_differ_passes():
    """보완 1-3: num_workers만 다르면 통과."""
    assert check_same_setup(record(), record(num_workers=8)) == []


def test_same_setup_env_differ_warns():
    """A11-2: 커밋·버전만 다르면 경고 목록."""
    aug = record(
        run_name="genimage_aug",
        augment=dict(AUG),
        git_commit="c2",
        torch_version="2.6.0",
        pillow_version="12.0.0",
        libjpeg_version="9.0",
        gpu_name="NVIDIA L4",
    )
    warnings_ = check_same_setup(record(), aug)
    assert len(warnings_) == 5
    for key in ("git_commit", "torch_version", "pillow_version", "libjpeg_version", "gpu_name"):
        assert any(w.startswith(key) for w in warnings_)


@pytest.mark.parametrize(
    "overrides, key",
    [
        ({"seed": 0}, "seed"),
        ({"manifest_sha256": "bbb"}, "manifest_sha256"),
        ({"train_generators": ["sd14", "sd15"]}, "train_generators"),
        ({"epochs": 1}, "epochs"),
        ({"lr": 1e-3}, "lr"),
        ({"crop_size": 256}, "crop_size"),
    ],
)
def test_same_setup_other_key_differ_raises(overrides, key):
    """A11-3~5: seed·manifest_sha256·train_generators(등) 다르면 에러, 다른 키 이름 출력."""
    aug = record(run_name="genimage_aug", augment=dict(AUG), **overrides)
    with pytest.raises(ConfigError) as exc:
        check_same_setup(record(), aug)
    assert key in str(exc.value)
    assert "run_name" not in str(exc.value) and "augment" not in str(exc.value)


def test_same_setup_key_only_on_one_side_raises():
    """한쪽에만 있는 키도 다른 것으로 본다 (예: 베이스라인엔 train_generators 없음)."""
    base = record()
    del base["train_generators"]
    with pytest.raises(ConfigError, match="train_generators"):
        check_same_setup(base, record(run_name="genimage_aug", augment=dict(AUG)))


def test_same_setup_smoke_vs_cifake_baseline_errors_on_epochs():
    """B4를 로컬로 미리 확인: cifake_aug_smoke는 epochs가 달라 에러 (run_name·augment는 허용)."""
    env = {k: "x" for k in rs.ENV_KEYS}
    base = {**base_config(), "manifest_sha256": "s", **env}
    smoke = {**load_config(CONFIGS_DIR / "cifake_aug_smoke.json"), "manifest_sha256": "s", **env}
    with pytest.raises(ConfigError) as exc:
        check_same_setup(base, smoke)
    assert "['epochs']" in str(exc.value)
