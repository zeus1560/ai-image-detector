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
