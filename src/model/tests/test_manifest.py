"""SPEC-01a 완료 기준 테스트: manifest.py와 dataset.py."""

import pytest
from PIL import Image

from conftest import make_row, valid_rows
from src.model import manifest as mf
from src.model.dataset import ManifestDataset
from src.model.manifest import LeakageError, ManifestValidationError


def leak_rows():
    """r0의 jpeg75 변형본을 test에 넣어 train/test 겹침을 만든다."""
    rows = valid_rows()
    for r in rows:
        if r["origin_id"] == "r0" and r["condition"] == "jpeg75":
            r["split"] = "test"
    return rows


# 완료 기준 1
def test_valid_manifest_passes_and_logs_zero_overlap(build, capsys):
    manifest_path, data_root = build(valid_rows())
    df = mf.load_manifest(manifest_path, data_root)
    mf.validate(df, data_root)
    out = capsys.readouterr().out
    assert "split 겹침 origin_id 0개" in out


# 완료 기준 2
def test_train_test_overlap_raises_leakage_and_no_dataset(build):
    manifest_path, data_root = build(leak_rows())
    df = mf.load_manifest(manifest_path, data_root)
    with pytest.raises(LeakageError) as exc:
        mf.validate(df, data_root)
    assert "r0" in str(exc.value)

    ds = None
    with pytest.raises(LeakageError):
        ds = ManifestDataset(manifest_path, data_root)
    assert ds is None


# 완료 기준 3
def test_train_filtered_dataset_still_detects_overlap(build):
    manifest_path, data_root = build(leak_rows())
    with pytest.raises(LeakageError):
        ManifestDataset(manifest_path, data_root, split="train")


# 완료 기준 4
@pytest.mark.parametrize("dropped", ["origin_id", "split", "label"])
def test_missing_required_column_names_column(build, dropped):
    columns = [c for c in mf.REQUIRED_COLUMNS if c != dropped]
    manifest_path, data_root = build(valid_rows(), columns=columns)
    with pytest.raises(ManifestValidationError) as exc:
        mf.load_manifest(manifest_path, data_root)
    assert dropped in str(exc.value)


# 처리 방식: load_manifest 허용값 검사 (열 이름과 행 번호)
def test_disallowed_value_reports_column_and_line(build):
    rows = valid_rows()
    rows[2]["label"] = "ai"  # DataFrame 인덱스 2 → CSV 4번 줄
    manifest_path, data_root = build(rows)
    with pytest.raises(ManifestValidationError) as exc:
        mf.load_manifest(manifest_path, data_root)
    msg = str(exc.value)
    assert "'label'" in msg
    assert "4번 줄" in msg


# 완료 기준 5 (label 불일치)
def test_label_mismatch_is_validation_error_not_leakage(build):
    rows = valid_rows()
    for r in rows:
        if r["origin_id"] == "f0" and r["condition"] == "jpeg75":
            r["label"] = "real"
    manifest_path, data_root = build(rows)
    df = mf.load_manifest(manifest_path, data_root)
    with pytest.raises(ManifestValidationError) as exc:
        mf.validate(df, data_root)
    assert exc.type is ManifestValidationError
    assert "f0" in str(exc.value)


# 완료 기준 5 (kakao 출처의 train 배정). 같은 규칙인 latest도 함께 확인
@pytest.mark.parametrize("source", ["kakao", "latest"])
def test_test_only_source_in_train_is_validation_error_not_leakage(build, source):
    rows = valid_rows()
    rows.append(make_row("x0", "real", "train", source=source, condition="original"))
    manifest_path, data_root = build(rows)
    df = mf.load_manifest(manifest_path, data_root)
    with pytest.raises(ManifestValidationError) as exc:
        mf.validate(df, data_root)
    assert exc.type is ManifestValidationError
    assert source in str(exc.value)


# 완료 기준 6
def test_overlap_and_label_mismatch_reported_together_as_leakage(build):
    rows = leak_rows()
    for r in rows:
        if r["origin_id"] == "f0" and r["condition"] == "jpeg75":
            r["label"] = "real"
    manifest_path, data_root = build(rows)
    df = mf.load_manifest(manifest_path, data_root)
    with pytest.raises(LeakageError) as exc:
        mf.validate(df, data_root)
    assert exc.type is LeakageError
    msg = str(exc.value)
    assert "split 겹침 origin_id 1개" in msg and "r0" in msg
    assert "label이 다른 ID 1개" in msg and "f0" in msg


# 완료 기준 7
def test_missing_file_reports_path(build):
    rows = valid_rows()
    gone = rows[5]["path"]
    manifest_path, data_root = build(rows, skip_files={gone})
    df = mf.load_manifest(manifest_path, data_root)
    with pytest.raises(ManifestValidationError) as exc:
        mf.validate(df, data_root, check_files="all")
    assert exc.type is ManifestValidationError
    assert gone in str(exc.value)


# 완료 기준 8
def test_sample_file_check_checks_exactly_n(build, monkeypatch, capsys):
    manifest_path, data_root = build(valid_rows())
    df = mf.load_manifest(manifest_path, data_root)

    checked = []
    real_exists = mf._file_exists

    def counting_exists(p):
        checked.append(p)
        return real_exists(p)

    monkeypatch.setattr(mf, "_file_exists", counting_exists)
    mf.validate(df, data_root, check_files=5)
    assert len(checked) == 5
    assert len(set(checked)) == 5
    assert "샘플 검사 5개" in capsys.readouterr().out

    # seed 고정: 다시 돌려도 같은 5개를 고른다
    first = list(checked)
    checked.clear()
    mf.validate(df, data_root, check_files=5)
    assert checked == first


# 완료 기준 9
def test_filter_count_matches_summary_cell(build):
    manifest_path, data_root = build(valid_rows())
    df = mf.load_manifest(manifest_path, data_root)
    table = mf.summary(df)
    for (label, condition), row in table.iterrows():
        for split, count in row.items():
            picked = mf.filter(df, condition=condition, split=split)
            assert (picked["label"] == label).sum() == count
    # 구체적인 칸 하나도 직접 확인
    assert table.loc[("real", "original"), "train"] == 2
    assert len(mf.filter(df, source="kakao", split="test")) == table.loc[("real", "kakao"), "test"]


# 완료 기준 10
def test_dataset_getitem_returns_image_label_row_index(build):
    rows = valid_rows()
    manifest_path, data_root = build(rows)
    ds = ManifestDataset(manifest_path, data_root, split="train")
    assert len(ds) == 8

    seen = set()
    for i in range(len(ds)):
        image, label, row_index = ds[i]
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB" and image.size == (8, 8)
        expected = {"real": 0, "fake": 1}[rows[row_index]["label"]]
        assert label == expected
        assert rows[row_index]["split"] == "train"
        seen.add(label)
    assert seen == {0, 1}


def test_dataset_converts_grayscale_to_rgb(build):
    rows = valid_rows()
    manifest_path, data_root = build(rows)
    # 첫 행 이미지를 흑백(L 모드)으로 덮어쓴다
    gray_path = data_root / rows[0]["path"]
    Image.new("L", (8, 8), 128).save(gray_path, format="PNG")
    with Image.open(gray_path) as im:
        assert im.mode == "L"

    ds = ManifestDataset(manifest_path, data_root)
    image, _, row_index = ds[0]
    assert row_index == 0
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (128, 128, 128)


def test_dataset_applies_external_transform(build):
    manifest_path, data_root = build(valid_rows())
    ds = ManifestDataset(manifest_path, data_root, transform=lambda im: im.size)
    image, _, _ = ds[0]
    assert image == (8, 8)
