"""SPEC-01a v2 완료 기준 테스트: color_mode 열, manifest sha256 해시.

v1 테스트(test_manifest.py)는 수정하지 않고 그대로 둔다 (v2 완료 기준 16).
"""

import hashlib
import re

import pytest

from conftest import valid_rows
from src.model import manifest as mf
from src.model.dataset import ManifestDataset
from src.model.manifest import ManifestValidationError

SHA_PATTERN = re.compile(r"sha256=([0-9a-f]{64})")


def printed_sha256(out: str) -> str:
    found = SHA_PATTERN.findall(out)
    assert len(found) == 1, f"출력에서 sha256을 하나만 찾아야 합니다: {out!r}"
    return found[0]


# v2 완료 기준 12: color_mode 열이 없으면 "color_mode"가 담긴 에러
def test_missing_color_mode_column_names_column(build):
    columns = [c for c in mf.REQUIRED_COLUMNS if c != "color_mode"]
    manifest_path, data_root = build(valid_rows(), columns=columns)
    with pytest.raises(ManifestValidationError) as exc:
        mf.load_manifest(manifest_path, data_root)
    assert "color_mode" in str(exc.value)


# v2 완료 기준 13: 허용값 밖의 color_mode → 열 이름과 줄 번호
def test_disallowed_color_mode_reports_column_and_line(build):
    rows = valid_rows()
    rows[3]["color_mode"] = "RGBA"  # DataFrame 인덱스 3 → CSV 5번 줄
    manifest_path, data_root = build(rows)
    with pytest.raises(ManifestValidationError) as exc:
        mf.load_manifest(manifest_path, data_root)
    msg = str(exc.value)
    assert "'color_mode'" in msg
    assert "5번 줄='RGBA'" in msg


@pytest.mark.parametrize("mode", ["RGB", "L", "CMYK"])
def test_allowed_color_modes_pass(build, mode):
    rows = valid_rows()
    for r in rows:
        r["color_mode"] = mode
    manifest_path, data_root = build(rows)
    df = mf.load_manifest(manifest_path, data_root)
    assert set(df["color_mode"]) == {mode}


# v2 완료 기준 14: 출력된 sha256 == hashlib 직접 계산값, 1바이트 변경 시 달라짐
def test_printed_sha256_matches_hashlib_and_changes_with_one_byte(build, capsys):
    manifest_path, data_root = build(valid_rows())
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    df = mf.load_manifest(manifest_path, data_root)
    out = capsys.readouterr().out
    assert str(manifest_path) in out
    assert f"행 {len(df)}개" in out
    printed = printed_sha256(out)
    assert printed == expected
    assert mf.manifest_sha256(manifest_path) == expected

    # 1바이트만 바꾼다: 첫 'sd14'를 'sd15'로 (여전히 유효한 manifest)
    data = bytearray(manifest_path.read_bytes())
    pos = data.index(b"sd14") + 3
    data[pos : pos + 1] = b"5"
    manifest_path.write_bytes(bytes(data))

    mf.load_manifest(manifest_path, data_root)
    changed = printed_sha256(capsys.readouterr().out)
    assert changed != printed
    assert changed == hashlib.sha256(bytes(data)).hexdigest()
    assert mf.manifest_sha256(manifest_path) == changed


# v2 완료 기준 15: Dataset.manifest_sha256 == load_manifest가 출력한 값
def test_dataset_manifest_sha256_matches_printed(build, capsys):
    manifest_path, data_root = build(valid_rows())
    ds = ManifestDataset(manifest_path, data_root, split="train")
    printed = printed_sha256(capsys.readouterr().out)
    assert ds.manifest_sha256 == printed
