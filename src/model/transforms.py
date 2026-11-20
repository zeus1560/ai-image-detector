"""크롭 좌표 계산과 PIL 크롭 (SPEC-02 처리 방식 1절).

torch를 import하지 않는다 (로컬 Windows에서 torch DLL이 차단되어 로컬 테스트를 가능하게 하기 위함).
텐서 변환과 정규화는 train.py에서 torchvision으로 한다.

규칙
- 리사이즈는 하지 않는다.
- 크롭 시작 좌표는 8의 배수 (기준본의 JPEG 8×8 블록 격자와 맞추기 위함).
- crop_size가 None이면 크롭하지 않고 이미지를 그대로 돌려준다 (CIFAKE 32×32).
"""

from __future__ import annotations

import numpy as np

# JPEG 블록 크기. 크롭 시작 좌표는 이 값의 배수로 맞춘다
GRID = 8


class CropSizeError(ValueError):
    """이미지의 한 변이라도 크롭 크기보다 작을 때 나는 에러."""


def _check_size(width: int, height: int, crop_size: int, path) -> None:
    if width < crop_size or height < crop_size:
        raise CropSizeError(
            f"이미지가 크롭 크기보다 작습니다: {width}×{height} < {crop_size} (path: {path})"
        )


def train_crop_origin(
    width: int, height: int, crop_size: int, seed: int, epoch: int, row_index: int, path=None
) -> tuple[int, int]:
    """학습 크롭 시작 좌표 (x, y)를 정한다.

    난수 생성기를 numpy.random.default_rng([seed, epoch, row_index])로 이미지마다 새로 만들기 때문에
    워커 수나 호출 순서와 상관없이 같은 (seed, epoch, row_index)면 같은 좌표가 나온다.
    x를 먼저, y를 나중에 뽑는다. 각각 {0, 8, ..., ⌊(W−C)/8⌋×8} 중 균등하게 고른다.
    """
    _check_size(width, height, crop_size, path)
    rng = np.random.default_rng([seed, epoch, row_index])
    x = GRID * int(rng.integers(0, (width - crop_size) // GRID + 1))
    y = GRID * int(rng.integers(0, (height - crop_size) // GRID + 1))
    return x, y


def eval_crop_origin(width: int, height: int, crop_size: int, path=None) -> tuple[int, int]:
    """평가(중앙) 크롭 시작 좌표 (x, y). 중앙 위치 ⌊(W−C)/2⌋를 8의 배수로 내린다.

    예: 512×512 → (128, 128), 300×260 → (16, 0), 256×256 → (0, 0)
    한 변이라도 크롭 크기보다 작으면 에러 (SPEC-02 미해결 1 결정 전 동작).
    """
    _check_size(width, height, crop_size, path)
    x = (width - crop_size) // 2 // GRID * GRID
    y = (height - crop_size) // 2 // GRID * GRID
    return x, y


def _crop(image, x: int, y: int, crop_size: int):
    return image.crop((x, y, x + crop_size, y + crop_size))


def train_crop(image, crop_size, seed: int, epoch: int, row_index: int, path=None):
    """PIL 이미지를 학습 규칙으로 자른다. crop_size가 None이면 이미지를 그대로 돌려준다."""
    if crop_size is None:
        return image
    x, y = train_crop_origin(image.width, image.height, crop_size, seed, epoch, row_index, path)
    return _crop(image, x, y, crop_size)


def eval_crop(image, crop_size, path=None):
    """PIL 이미지를 평가(중앙) 규칙으로 자른다. crop_size가 None이면 이미지를 그대로 돌려준다."""
    if crop_size is None:
        return image
    x, y = eval_crop_origin(image.width, image.height, crop_size, path)
    return _crop(image, x, y, crop_size)


CROP_MODES = ("train", "eval")


class CropWrapper:
    """ManifestDataset을 감싸 크롭과 텐서 변환을 적용한다 (SPEC-02 확정 후 보완 2-1).

    ManifestDataset의 transform은 PIL 이미지만 받아 epoch·row_index를 쓸 수 없으므로,
    train.py는 ManifestDataset을 transform=None으로 만들고 이 클래스로 감싼다.

    Args:
        dataset: (PIL 이미지, label, row_index)를 돌려주고 df(path 열 포함, 인덱스 = row_index)를 가진 Dataset
        crop_size: 크롭 크기 (None이면 크롭하지 않음)
        mode: "train"(train_crop) 또는 "eval"(eval_crop)
        seed: 학습 크롭 난수 seed
        to_tensor: 크롭된 PIL 이미지를 받아 변환하는 함수 (train.py가 ToTensor + Normalize를 넣는다)

    mode="train"은 set_epoch(epoch)를 부른 뒤에만 꺼낼 수 있다 (epoch 기본값으로 조용히 도는 것을 막기 위함).
    DataLoader는 persistent_workers=False로 써야 한다. True면 워커가 첫 epoch 때 복사한 epoch 값을 계속 쓴다.
    """

    def __init__(self, dataset, crop_size, mode: str, seed: int, to_tensor):
        if mode not in CROP_MODES:
            raise ValueError(f"mode는 {CROP_MODES} 중 하나여야 합니다: {mode!r}")
        self.dataset = dataset
        self.crop_size = crop_size
        self.mode = mode
        self.seed = seed
        self.to_tensor = to_tensor
        self.epoch = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i):
        if self.mode == "train" and self.epoch is None:
            raise RuntimeError("train 모드 CropWrapper는 set_epoch(epoch)를 먼저 불러야 합니다")
        image, label, row_index = self.dataset[i]
        # path는 크롭 에러 메시지용이라 크롭할 때만 꺼낸다
        path = None if self.crop_size is None else self.dataset.df.loc[row_index, "path"]
        if self.mode == "train":
            image = train_crop(image, self.crop_size, self.seed, self.epoch, row_index, path)
        else:
            image = eval_crop(image, self.crop_size, path)
        return self.to_tensor(image), label, row_index
