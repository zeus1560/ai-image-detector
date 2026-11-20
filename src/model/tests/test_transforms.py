"""SPEC-02 A: transforms.py 테스트 (torch 없이)."""

import random

import pytest
from PIL import Image

from src.model.transforms import (
    CropSizeError,
    eval_crop,
    eval_crop_origin,
    train_crop,
    train_crop_origin,
)


def gradient_image(width, height):
    """픽셀마다 값이 달라 어느 위치를 잘랐는지 구분할 수 있는 이미지."""
    img = Image.new("RGB", (width, height))
    img.putdata([(x % 256, y % 256, (x + y) % 256) for y in range(height) for x in range(width)])
    return img


# ---- 학습 크롭 ----

def test_train_crop_coords_on_grid_and_in_range():
    """A1: (seed, epoch, row_index)를 바꿔도 좌표가 8의 배수, 0~256 범위, 결과 256×256."""
    img = gradient_image(512, 512)
    xs, ys = set(), set()
    for seed in (0, 1, 42):
        for epoch in (1, 2, 3):
            for row_index in range(30):
                x, y = train_crop_origin(512, 512, 256, seed, epoch, row_index)
                assert x % 8 == 0 and y % 8 == 0
                assert 0 <= x <= 256 and 0 <= y <= 256
                xs.add(x)
                ys.add(y)
                out = train_crop(img, 256, seed, epoch, row_index)
                assert out.size == (256, 256)
                # 계산한 좌표의 영역을 그대로 잘랐는지 확인
                assert out.tobytes() == img.crop((x, y, x + 256, y + 256)).tobytes()
    # 좌표가 한 값에 고정되지 않고 여러 위치가 나온다
    assert len(xs) > 1 and len(ys) > 1


def test_train_crop_deterministic_per_key_regardless_of_order():
    """A2: 같은 (seed, epoch, row_index)면 같은 좌표, 호출 순서를 섞어도 같다."""
    keys = [(seed, epoch, row) for seed in (0, 42) for epoch in (1, 2) for row in range(20)]
    first = {k: train_crop_origin(512, 512, 256, *k) for k in keys}
    second = {k: train_crop_origin(512, 512, 256, *k) for k in keys}
    assert first == second

    shuffled = keys[:]
    random.Random(7).shuffle(shuffled)
    third = {k: train_crop_origin(512, 512, 256, *k) for k in shuffled}
    assert first == third


def test_train_crop_too_small_raises_with_path():
    """A3: 한 변이 C보다 작으면 에러, 메시지에 path가 있다."""
    img = gradient_image(200, 512)
    with pytest.raises(CropSizeError, match="images/genimage/x.jpg"):
        train_crop(img, 256, 42, 1, 0, path="images/genimage/x.jpg")


# ---- 평가 크롭 ----

@pytest.mark.parametrize(
    "size, expected",
    [((512, 512), (128, 128)), ((300, 260), (16, 0)), ((256, 256), (0, 0))],
)
def test_eval_crop_origin(size, expected):
    """A4: 512×512 → (128,128), 300×260 → (16,0), 256×256 → (0,0)."""
    assert eval_crop_origin(*size, 256) == expected
    img = gradient_image(*size)
    out = eval_crop(img, 256)
    x, y = expected
    assert out.size == (256, 256)
    assert out.tobytes() == img.crop((x, y, x + 256, y + 256)).tobytes()


def test_eval_crop_too_small_raises():
    """A5: 한 변이 C보다 작으면 에러 (미해결 1 결정 전 동작)."""
    img = gradient_image(200, 512)
    with pytest.raises(CropSizeError, match="images/genimage/y.jpg"):
        eval_crop(img, 256, path="images/genimage/y.jpg")


# ---- crop_size null ----

def test_null_crop_returns_image_unchanged():
    """A6: crop_size가 None이면 32×32 이미지가 바이트·크기 변화 없이 그대로 나온다."""
    img = gradient_image(32, 32)
    before = img.tobytes()
    for out in (train_crop(img, None, 42, 1, 0), eval_crop(img, None)):
        assert out.size == (32, 32)
        assert out.tobytes() == before


# ---- CropWrapper (확정 후 보완 2-1) ----

import pandas as pd  # noqa: E402

from src.model.transforms import CropWrapper  # noqa: E402


class FakeDataset:
    """ManifestDataset 대신 쓰는 가짜 dataset: (PIL, label, row_index)를 돌려주고 df.path를 갖는다.

    row_index를 위치(0, 1, 2…)와 다르게 두어 CropWrapper가 위치가 아니라 row_index를 쓰는지 확인한다.
    """

    def __init__(self, sizes, row_indices):
        self.df = pd.DataFrame(
            {"path": [f"images/fake/{r}.png" for r in row_indices]}, index=row_indices
        )
        self.images = [gradient_image(w, h) for w, h in sizes]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], i % 2, int(self.df.index[i])


def identity(img):
    return img


def make_wrapper(mode="train", crop_size=256, seed=42, sizes=None, row_indices=(10, 20, 30, 40)):
    sizes = sizes or [(512, 512)] * len(row_indices)
    return CropWrapper(FakeDataset(sizes, list(row_indices)), crop_size, mode, seed, identity)


def test_wrapper_same_coords_regardless_of_order():
    """같은 (seed, epoch, row_index)는 꺼내는 순서와 상관없이 같은 좌표(같은 픽셀)."""
    w = make_wrapper()
    w.set_epoch(3)
    forward = {i: w[i][0].tobytes() for i in range(len(w))}
    order = list(range(len(w)))
    random.Random(0).shuffle(order)
    shuffled = {i: w[i][0].tobytes() for i in reversed(order)}
    assert forward == shuffled
    # 기대값: train_crop을 row_index로 직접 호출한 결과와 같다 (위치 i가 아니라 row_index 사용)
    for i in range(len(w)):
        img, label, row_index = w[i]
        assert row_index == w.dataset.df.index[i]
        assert label == i % 2
        expected = train_crop(w.dataset.images[i], 256, 42, 3, row_index)
        assert img.tobytes() == expected.tobytes()


def test_wrapper_set_epoch_changes_coords():
    """set_epoch로 epoch를 바꾸면 좌표가 바뀐다 (persistent_workers 문제를 로컬에서 대신 확인)."""
    w = make_wrapper(row_indices=tuple(range(100, 120)))
    w.set_epoch(1)
    e1 = [w[i][0].tobytes() for i in range(len(w))]
    w.set_epoch(2)
    e2 = [w[i][0].tobytes() for i in range(len(w))]
    assert e1 != e2
    # epoch 1로 되돌리면 처음과 같다
    w.set_epoch(1)
    assert [w[i][0].tobytes() for i in range(len(w))] == e1


def test_wrapper_train_without_set_epoch_raises():
    """set_epoch 전 train 모드 호출은 에러 (crop_size=None이어도)."""
    for crop_size in (256, None):
        w = make_wrapper(crop_size=crop_size)
        with pytest.raises(RuntimeError, match="set_epoch"):
            w[0]


def test_wrapper_eval_needs_no_epoch_and_uses_center():
    w = make_wrapper(mode="eval")
    img, _, _ = w[0]
    assert img.tobytes() == eval_crop(w.dataset.images[0], 256).tobytes()


def test_wrapper_null_crop_returns_image_unchanged():
    """crop_size=None이면 32×32 이미지가 그대로 나온다 (train·eval 모두)."""
    for mode in ("train", "eval"):
        w = make_wrapper(mode=mode, crop_size=None, sizes=[(32, 32)] * 4)
        w.set_epoch(1)
        for i in range(len(w)):
            img, _, _ = w[i]
            assert img.size == (32, 32)
            assert img.tobytes() == w.dataset.images[i].tobytes()


def test_wrapper_crop_error_shows_path_from_df():
    """크롭 에러 메시지의 path는 dataset.df에서 꺼낸다."""
    w = make_wrapper(sizes=[(200, 512)] * 4)
    w.set_epoch(1)
    with pytest.raises(CropSizeError, match="images/fake/30.png"):
        w[2]


def test_wrapper_with_real_manifest_dataset_uses_row_path(build):
    """진짜 ManifestDataset(split="val"로 필터링)과 붙었을 때 row_index로 맞는 path를 꺼낸다."""
    from conftest import valid_rows

    from src.model.dataset import ManifestDataset

    rows = valid_rows()  # train·val·test, original·jpeg75, kakao가 섞인 8×8 이미지
    manifest_path, data_root = build(rows)
    ds = ManifestDataset(manifest_path, data_root, split="val")
    val_indices = [i for i, r in enumerate(rows) if r["split"] == "val"]
    # 필터링 후 위치(0, 1, …)와 manifest 행 인덱스가 달라야 이 테스트가 의미 있다
    assert list(ds.df.index) == val_indices
    assert val_indices != list(range(len(val_indices)))

    w = CropWrapper(ds, None, "eval", 0, identity)
    assert len(w) == len(val_indices)
    for i in range(len(w)):
        img, label, row_index = w[i]
        assert row_index == val_indices[i]
        assert w.dataset.df.loc[row_index, "path"] == rows[row_index]["path"]
        assert label == (0 if rows[row_index]["label"] == "real" else 1)
        assert img.size == (8, 8)

    # 이미지(8×8)보다 큰 crop_size: 에러 메시지에 그 행의 path가 들어간다 (train·eval 모두)
    for mode in ("train", "eval"):
        w = CropWrapper(ds, 16, mode, 0, identity)
        w.set_epoch(1)
        for i in range(len(w)):
            expected_path = rows[val_indices[i]]["path"]
            with pytest.raises(CropSizeError) as exc:
                w[i]
            assert expected_path in str(exc.value)


def test_wrapper_applies_to_tensor_and_rejects_bad_mode():
    w = CropWrapper(FakeDataset([(32, 32)], [5]), None, "eval", 0, lambda img: ("T", img.size))
    assert w[0][0] == ("T", (32, 32))
    with pytest.raises(ValueError):
        CropWrapper(FakeDataset([(32, 32)], [5]), None, "test", 0, identity)
