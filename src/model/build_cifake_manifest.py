"""CIFAKE를 데이터 계약 형식으로 바꾼다 (SPEC-01b).

원본 폴더(train/REAL, train/FAKE, test/REAL, test/FAKE)의 이미지를
출력 폴더의 images/cifake/original/{label}/{origin_id}.jpg 로 바이트 그대로 복사하고,
cifake_manifest.csv를 만든 뒤 SPEC-01a의 load_manifest·validate·summary를 실행한다.

실행 (레포 루트에서):
    python -m src.model.build_cifake_manifest <원본 폴더> <출력 폴더>
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from . import manifest as mf

# split 무작위 배정에 쓰는 고정 seed (SPEC-01b 처리 방식 5)
SPLIT_SEED = 42

# 원래 train에서 label별로 val로 떼어낼 장수
VAL_PER_LABEL = 10_000

MANIFEST_NAME = "cifake_manifest.csv"

# 원본 폴더 이름 → 계약의 label 값
LABEL_DIRS = {"REAL": "real", "FAKE": "fake"}
ORIGINAL_SPLITS = ("train", "test")
GENERATOR = {"real": "real", "fake": "sd14"}  # 논문 기준 Stable Diffusion 1.4

# "번호.jpg" 또는 "번호 (n).jpg"
FILENAME_RE = re.compile(r"^(\d+)(?: \((\d+)\))?\.jpg$")

# PIL format 이름 → 계약의 format 값
PIL_FORMAT = {"JPEG": "jpg", "PNG": "png"}


class CifakeBuildError(Exception):
    """CIFAKE manifest를 만들 수 없을 때 나는 에러."""


def parse_filename(name: str) -> tuple[int, int]:
    """파일 이름을 (번호, 순번)으로 해석한다. 괄호가 없으면 순번 1.

    규칙에 맞지 않으면 CifakeBuildError를 낸다.
    """
    m = FILENAME_RE.match(name)
    if m is None:
        raise CifakeBuildError(f"파일 이름 규칙에 맞지 않습니다: {name!r}")
    number = int(m.group(1))
    seq = int(m.group(2)) if m.group(2) is not None else 1
    return number, seq


def make_origin_id(label: str, original_split: str, number: int, seq: int) -> str:
    return f"cifake_{label}_{original_split}_{number}_{seq}"


def collect_entries(src_root) -> list[dict]:
    """원본 폴더 4개를 훑어 파일마다 origin_id를 만든다.

    규칙에 맞지 않는 파일 이름이나 중복 origin_id가 있으면 모아서 한 번에 에러를 낸다.
    """
    src_root = Path(src_root)
    entries = []
    bad_names = []
    for original_split in ORIGINAL_SPLITS:
        for dir_name, label in LABEL_DIRS.items():
            folder = src_root / original_split / dir_name
            if not folder.is_dir():
                raise CifakeBuildError(f"원본 폴더가 없습니다: {folder}")
            for p in sorted(folder.iterdir()):
                try:
                    number, seq = parse_filename(p.name)
                except CifakeBuildError:
                    bad_names.append(str(p.relative_to(src_root)))
                    continue
                entries.append(
                    {
                        "src": p,
                        "label": label,
                        "original_split": original_split,
                        "origin_id": make_origin_id(label, original_split, number, seq),
                    }
                )
    if bad_names:
        raise CifakeBuildError(
            f"파일 이름 규칙(번호.jpg 또는 번호 (n).jpg)에 맞지 않는 파일 {len(bad_names)}개: "
            + ", ".join(bad_names)
        )

    by_id: dict[str, list[str]] = {}
    for e in entries:
        by_id.setdefault(e["origin_id"], []).append(str(e["src"].relative_to(src_root)))
    dups = {oid: files for oid, files in by_id.items() if len(files) > 1}
    if dups:
        examples = "; ".join(
            f"{oid} ← {', '.join(files)}" for oid, files in list(dups.items())[: mf.MAX_EXAMPLES]
        )
        raise CifakeBuildError(f"origin_id 중복 {len(dups)}개. 예: {examples}")

    entries.sort(key=lambda e: e["origin_id"])
    return entries


def assign_splits(entries: list[dict], seed: int = SPLIT_SEED, val_per_label: int = VAL_PER_LABEL) -> None:
    """원래 test → test. 원래 train은 label별로 val_per_label장을 무작위로 val, 나머지 train.

    entries는 origin_id 순으로 정렬되어 있어야 같은 seed에서 같은 결과가 나온다.
    """
    rng = random.Random(seed)
    for e in entries:
        e["split"] = "test" if e["original_split"] == "test" else "train"
    for label in sorted(LABEL_DIRS.values()):
        pool = [e for e in entries if e["original_split"] == "train" and e["label"] == label]
        if len(pool) < val_per_label:
            raise CifakeBuildError(
                f"원래 train의 {label} 이미지가 {len(pool)}장으로 val {val_per_label}장보다 적습니다"
            )
        for e in rng.sample(pool, val_per_label):
            e["split"] = "val"


def read_image_info(path: Path) -> dict:
    """실제 파일을 PIL로 열어 width, height, format, color_mode를 읽는다."""
    with Image.open(path) as img:
        fmt = PIL_FORMAT.get(img.format)
        if fmt is None:
            raise CifakeBuildError(f"지원하지 않는 이미지 포맷 {img.format!r}: {path}")
        return {
            "width": str(img.width),
            "height": str(img.height),
            "format": fmt,
            "color_mode": img.mode,
        }


def build(src_root, out_root, seed: int = SPLIT_SEED, val_per_label: int = VAL_PER_LABEL) -> Path:
    """이미지를 복사하고 cifake_manifest.csv를 쓴 뒤 검증한다. manifest 경로를 반환한다."""
    src_root = Path(src_root).resolve()
    out_root = Path(out_root).resolve()
    # 원본을 덮어쓰지 않도록 출력 폴더가 원본과 겹치지 않는지 확인한다
    if out_root == src_root or src_root in out_root.parents or out_root in src_root.parents:
        raise CifakeBuildError(f"출력 폴더가 원본 폴더와 겹칩니다: 원본={src_root}, 출력={out_root}")

    print(f"원본 폴더: {src_root}")
    print(f"출력 폴더: {out_root}")
    print(f"split seed: {seed}, label별 val 장수: {val_per_label}")

    entries = collect_entries(src_root)
    n_source_files = sum(
        1
        for s in ORIGINAL_SPLITS
        for d in LABEL_DIRS
        for p in (src_root / s / d).iterdir()
        if p.is_file()
    )
    print(f"원본 파일 수: {n_source_files}")

    assign_splits(entries, seed=seed, val_per_label=val_per_label)

    rows = []
    for e in entries:
        rel = f"images/cifake/original/{e['label']}/{e['origin_id']}.jpg"
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # PIL로 다시 저장하면 JPEG가 재압축되므로 바이트 그대로 복사한다
        shutil.copyfile(e["src"], dst)
        info = read_image_info(dst)
        rows.append(
            {
                "path": rel,
                "label": e["label"],
                "source": "cifake",
                "generator": GENERATOR[e["label"]],
                "origin_id": e["origin_id"],
                "condition": "original",
                "split": e["split"],
                **info,
            }
        )

    manifest_path = out_root / MANIFEST_NAME
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mf.REQUIRED_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    df_rows = pd.DataFrame(rows)
    print(f"manifest 행 수: {len(df_rows)} (원본 파일 수와 {'같음' if len(df_rows) == n_source_files else '다름'})")
    print("[장수표] label × split")
    print(pd.crosstab(df_rows["label"], df_rows["split"]).to_string())
    for col in ("color_mode", "width", "height"):
        print(f"[전수 분포] label별 {col}")
        print(pd.crosstab(df_rows["label"], df_rows[col]).to_string())

    df = mf.load_manifest(manifest_path, out_root)
    mf.validate(df, out_root, check_files="all")
    mf.summary(df)
    return manifest_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="CIFAKE manifest 만들기 (SPEC-01b)")
    parser.add_argument("src_root", help="CIFAKE 원본 폴더 (train/REAL 등이 있는 곳)")
    parser.add_argument("out_root", help="출력 폴더 (계약의 data/ 역할)")
    args = parser.parse_args(argv)
    build(args.src_root, args.out_root)


if __name__ == "__main__":
    main()
