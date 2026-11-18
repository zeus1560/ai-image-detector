"""SPEC-01b 완료 기준 A: CIFAKE와 같은 이름 규칙의 미니 폴더로 확인한다."""

import pandas as pd
import pytest
from PIL import Image

from src.model import manifest as mf
from src.model.build_cifake_manifest import (
    MANIFEST_NAME,
    CifakeBuildError,
    build,
    parse_filename,
)

# 폴더별 파일 이름 (실제 CIFAKE의 폴더별 번호 표기를 흉내낸다)
# train/REAL과 train/FAKE는 "0000.jpg", "0000 (2).jpg"가 겹친다
MINI_LAYOUT = {
    ("train", "REAL"): ["0000.jpg", "0000 (2).jpg", "0000 (10).jpg", "0001.jpg", "0002.jpg"],
    ("train", "FAKE"): ["0000.jpg", "0000 (2).jpg", "1000.jpg", "1000 (3).jpg", "5999.jpg"],
    ("test", "REAL"): ["0000.jpg", "0000 (2).jpg", "0003.jpg"],
    ("test", "FAKE"): ["0.jpg", "0 (2).jpg", "7.jpg"],
}
MINI_VAL_PER_LABEL = 2


def make_mini_cifake(root, layout=MINI_LAYOUT):
    for i, ((split, label_dir), names) in enumerate(layout.items()):
        folder = root / split / label_dir
        folder.mkdir(parents=True)
        for j, name in enumerate(names):
            color = (10 * i + j, 40, 200 - 10 * j)
            Image.new("RGB", (8, 8), color).save(folder / name, format="JPEG", quality=75)
    return root


def run_build(tmp_path, out_name="out", layout=MINI_LAYOUT, seed=42):
    src = tmp_path / "cifake"
    if not src.exists():
        make_mini_cifake(src, layout)
    out = tmp_path / out_name
    manifest_path = build(src, out, seed=seed, val_per_label=MINI_VAL_PER_LABEL)
    return src, out, manifest_path


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# A-1
@pytest.mark.parametrize(
    "name, expected",
    [
        ("0000 (10).jpg", (0, 10)),
        ("0000.jpg", (0, 1)),
        ("0 (2).jpg", (0, 2)),
    ],
)
def test_parse_filename(name, expected):
    assert parse_filename(name) == expected


# A-2
def test_same_filename_in_real_and_fake_gets_different_origin_id(tmp_path):
    _, _, manifest_path = run_build(tmp_path)
    df = read_csv(manifest_path)
    real = df[(df["label"] == "real") & df["origin_id"].str.contains("_train_0_1$")]
    fake = df[(df["label"] == "fake") & df["origin_id"].str.contains("_train_0_1$")]
    assert real["origin_id"].tolist() == ["cifake_real_train_0_1"]
    assert fake["origin_id"].tolist() == ["cifake_fake_train_0_1"]
    assert df["origin_id"].is_unique


# A-3
def test_bad_filename_raises_with_name(tmp_path):
    layout = dict(MINI_LAYOUT)
    layout[("train", "REAL")] = MINI_LAYOUT[("train", "REAL")] + ["0004 copy.jpg"]
    with pytest.raises(CifakeBuildError, match="0004 copy.jpg"):
        run_build(tmp_path, layout=layout)


# A-4
def test_duplicate_origin_id_raises(tmp_path):
    layout = dict(MINI_LAYOUT)
    # "0000.jpg"와 "0.jpg"는 둘 다 번호 0·순번 1 → cifake_fake_test_0_1
    layout[("test", "FAKE")] = MINI_LAYOUT[("test", "FAKE")] + ["0000.jpg"]
    with pytest.raises(CifakeBuildError, match="origin_id 중복.*cifake_fake_test_0_1"):
        run_build(tmp_path, layout=layout)


# A-5
def test_copied_files_are_byte_identical(tmp_path):
    src, out, manifest_path = run_build(tmp_path)
    df = read_csv(manifest_path)
    row = df[df["origin_id"] == "cifake_real_train_0_10"].iloc[0]
    assert (out / row["path"]).read_bytes() == (src / "train" / "REAL" / "0000 (10).jpg").read_bytes()
    # 전체 파일도 확인: 원본 바이트 집합과 복사본 바이트 집합이 같다
    src_bytes = sorted(p.read_bytes() for p in src.rglob("*.jpg"))
    out_bytes = sorted((out / p).read_bytes() for p in df["path"])
    assert src_bytes == out_bytes


# A-6
def test_same_seed_gives_same_split(tmp_path):
    _, _, m1 = run_build(tmp_path, out_name="out1", seed=123)
    _, _, m2 = run_build(tmp_path, out_name="out2", seed=123)
    df1, df2 = read_csv(m1), read_csv(m2)
    assert df1["origin_id"].tolist() == df2["origin_id"].tolist()
    assert df1["split"].tolist() == df2["split"].tolist()
    # val이 실제로 무작위 배정되었는지 (label별 MINI_VAL_PER_LABEL장)
    assert (df1[df1["split"] == "val"]["label"].value_counts() == MINI_VAL_PER_LABEL).all()


# A-7
def test_result_passes_spec01a_validate(tmp_path):
    _, out, manifest_path = run_build(tmp_path)
    assert manifest_path.name == MANIFEST_NAME
    df = mf.load_manifest(manifest_path, out)
    mf.validate(df, out, check_files="all")  # 위반이 있으면 에러가 난다
    assert set(df["format"]) == {"jpg"}
    assert set(df["color_mode"]) == {"RGB"}
    assert set(df["split"]) == {"train", "val", "test"}
