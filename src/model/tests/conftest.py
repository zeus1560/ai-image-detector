"""SPEC-01a 테스트 공용 도구.

임시 폴더에 8×8 가짜 이미지와 작은 manifest CSV를 만든다 (CIFAKE 불필요).
"""

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

# 레포 루트를 import 경로에 넣어 `src.model`을 불러올 수 있게 한다
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.manifest import REQUIRED_COLUMNS  # noqa: E402


def make_row(origin_id, label, split, source="cifake", condition="original", fmt="png"):
    generator = "real" if label == "real" else "sd14"
    path = f"images/{source}/{condition}/{label}/{origin_id}_{condition}.{fmt}"
    return {
        "path": path,
        "label": label,
        "source": source,
        "generator": generator,
        "origin_id": origin_id,
        "condition": condition,
        "split": split,
        "width": "8",
        "height": "8",
        "format": fmt,
    }


def valid_rows():
    """겹침 없는 정상 manifest 행들 (원본 + jpeg75 변형본, kakao test 포함)."""
    rows = []
    for oid, label, split in [
        ("r0", "real", "train"),
        ("r1", "real", "train"),
        ("r2", "real", "val"),
        ("r3", "real", "test"),
        ("f0", "fake", "train"),
        ("f1", "fake", "train"),
        ("f2", "fake", "val"),
        ("f3", "fake", "test"),
    ]:
        rows.append(make_row(oid, label, split, condition="original"))
        rows.append(make_row(oid, label, split, condition="jpeg75"))
    rows.append(make_row("k0", "real", "test", source="kakao", condition="kakao"))
    return rows


@pytest.fixture
def build(tmp_path):
    """rows로 data_root와 manifest.csv를 만든다. skip_files의 path는 이미지를 만들지 않는다.

    반환: (manifest_path, data_root)
    """

    def _build(rows, columns=None, skip_files=()):
        data_root = tmp_path / "data"
        data_root.mkdir(exist_ok=True)
        columns = columns or REQUIRED_COLUMNS
        for i, r in enumerate(rows):
            if r["path"] in skip_files:
                continue
            p = data_root / r["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            color = (200, 30, 30) if r["label"] == "real" else (30, 30, 200)
            Image.new("RGB", (8, 8), color).save(p, format="PNG")
        manifest_path = data_root / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return manifest_path, data_root

    return _build
