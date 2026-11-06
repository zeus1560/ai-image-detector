"""manifest 기반 Dataset (SPEC-01a).

torch를 import하지 않는다. __len__과 __getitem__만 갖춘 클래스라서
PyTorch DataLoader에 그대로 넘길 수 있다.
(로컬 Windows에서 torch DLL이 차단되어 로컬 테스트를 가능하게 하기 위함)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from . import manifest as mf

LABEL_TO_INDEX = {"real": 0, "fake": 1}


class ManifestDataset:
    """manifest.csv의 행을 (이미지, label, 행 인덱스)로 돌려주는 Dataset.

    생성할 때 필터 전 전체 manifest에 validate를 실행한다.
    통과하지 못하면 에러가 나고 Dataset이 생성되지 않는다.

    Args:
        manifest_path: 데이터 계약 형식의 manifest CSV 경로 (전체 manifest)
        data_root: manifest의 path가 기준으로 삼는 data/ 폴더 위치
        source, condition, split: 고를 조건 (None이면 거르지 않음)
        transform: PIL 이미지를 받아 변환하는 함수 (None이면 PIL 이미지 그대로)
        check_files: validate에 넘길 파일 검사 방식 ("all" 또는 정수 n)
    """

    def __init__(
        self,
        manifest_path,
        data_root,
        source=None,
        condition=None,
        split=None,
        transform=None,
        check_files="all",
    ):
        self.data_root = Path(data_root)
        full = mf.load_manifest(manifest_path, data_root)
        # 필터 전 전체 manifest를 검사해야 split 간 겹침을 발견할 수 있다
        mf.validate(full, data_root, check_files=check_files)
        self.df = mf.filter(full, source=source, condition=condition, split=split)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.data_root / row["path"]) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = LABEL_TO_INDEX[row["label"]]
        row_index = int(self.df.index[i])
        return image, label, row_index
