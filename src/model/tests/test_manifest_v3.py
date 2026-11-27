"""SPEC-01a v3 완료 기준 테스트: gray 선택 열, 흑백 실제의 train/val 배정 검사, gray 장수표.

conftest.py와 v1·v2 테스트는 고치지 않는다 (gray 없는 픽스처가 그대로 통과해야 함).
gray 열이 필요한 manifest는 이 파일 안에서 행에 gray를 붙여 만든다.
"""

import hashlib

import pytest

from conftest import valid_rows
from src.model import manifest as mf
from src.model.dataset import ManifestDataset
from src.model.manifest import LeakageError, ManifestValidationError

GRAY_COLUMNS = mf.REQUIRED_COLUMNS + ["gray"]


def gray_rows(gray_real_test=True):
    """valid_rows에 gray 열을 붙인다. r3(실제, test) 행들만 True로 둘 수 있다."""
    rows = valid_rows()
    for r in rows:
        r["gray"] = "True" if (gray_real_test and r["origin_id"] == "r3") else "False"
    return rows


# v3-1: gray 열이 없으면 오류 없이 읽히고, 전부 False, 로그에 "gray 열 없음"
def test_absent_gray_column_fills_false_and_logs(build, capsys):
    rows = valid_rows()
    manifest_path, data_root = build(rows)
    df = mf.load_manifest(manifest_path, data_root)
    out = capsys.readouterr().out
    assert "gray" in df.columns
    assert df["gray"].dtype == bool
    assert not df["gray"].any()
    assert f"gray 열 없음 → 전부 False (행 {len(rows)}개)" in out
    assert df.attrs["gray_column"] == "absent"
    mf.validate(df, data_root)  # gray가 전부 False면 5번 위반 없음


# v3-2: gray 열 없는 manifest를 읽기 전후로 파일 sha256이 같다
def test_absent_gray_column_does_not_touch_file(build, capsys):
    manifest_path, data_root = build(valid_rows())
    before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    df = mf.load_manifest(manifest_path, data_root)
    after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert before == after
    assert df.attrs["manifest_sha256"] == before
    assert f"sha256={before}" in capsys.readouterr().out


# v3-3: gray가 'True'/'False'이면 df["gray"]가 bool이고 값이 파일과 같다
def test_present_gray_column_parsed_as_bool(build):
    rows = gray_rows()
    manifest_path, data_root = build(rows, columns=GRAY_COLUMNS)
    df = mf.load_manifest(manifest_path, data_root)
    assert df["gray"].dtype == bool
    assert df["gray"].tolist() == [r["gray"] == "True" for r in rows]
    assert df["gray"].sum() == 2  # r3의 original, jpeg75
    assert df.attrs["gray_column"] == "present"


# v3-4: gray 빈 칸 → ManifestValidationError, "gray"·줄 번호·빈 값 표시. False로 읽히지 않는다
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_gray_is_error_not_false(build, blank):
    rows = gray_rows()
    rows[4]["gray"] = blank  # DataFrame 인덱스 4 → CSV 6번 줄
    manifest_path, data_root = build(rows, columns=GRAY_COLUMNS)
    with pytest.raises(ManifestValidationError) as exc:
        mf.load_manifest(manifest_path, data_root)
    msg = str(exc.value)
    assert not isinstance(exc.value, LeakageError)
    assert "'gray'" in msg
    assert "빈 값" in msg
    assert "6번 줄" in msg


# v3-5: 'true', 'TRUE', '1', ' True' (와 'nan') → 열 이름과 줄 번호가 담긴 허용값 에러
@pytest.mark.parametrize("value", ["true", "TRUE", "1", " True", "nan"])
def test_disallowed_gray_value_reports_column_and_line(build, value):
    rows = gray_rows()
    rows[2]["gray"] = value  # CSV 4번 줄
    manifest_path, data_root = build(rows, columns=GRAY_COLUMNS)
    with pytest.raises(ManifestValidationError) as exc:
        mf.load_manifest(manifest_path, data_root)
    msg = str(exc.value)
    assert "'gray'에 허용되지 않은 값 1개" in msg
    assert f"4번 줄='{value}'" in msg
    assert "빈 값" not in msg


# v3-6: 빈 값과 허용값 밖 값이 함께 있으면 에러 하나에 두 위반이 모두 나온다 (기존 허용값 위반과도 함께)
def test_blank_and_disallowed_gray_reported_together(build):
    rows = gray_rows()
    rows[1]["gray"] = ""  # 3번 줄
    rows[5]["gray"] = "true"  # 7번 줄
    rows[6]["color_mode"] = "RGBA"  # 8번 줄, 기존 허용값 위반
    manifest_path, data_root = build(rows, columns=GRAY_COLUMNS)
    with pytest.raises(ManifestValidationError) as exc:
        mf.load_manifest(manifest_path, data_root)
    msg = str(exc.value)
    assert "'gray'에 빈 값 1개" in msg and "3번 줄" in msg
    assert "'gray'에 허용되지 않은 값 1개" in msg and "7번 줄='true'" in msg
    assert "'color_mode'" in msg and "8번 줄='RGBA'" in msg


# v3-7: real·gray=True가 train(또는 val)에 있으면 ManifestValidationError, LeakageError 아님
@pytest.mark.parametrize("oid,split", [("r0", "train"), ("r2", "val")])
def test_gray_real_in_train_or_val_is_error(build, oid, split):
    rows = gray_rows()
    for r in rows:
        if r["origin_id"] == oid:
            r["gray"] = "True"
    manifest_path, data_root = build(rows, columns=GRAY_COLUMNS)
    df = mf.load_manifest(manifest_path, data_root)
    with pytest.raises(ManifestValidationError) as exc:
        mf.validate(df, data_root)
    assert not isinstance(exc.value, LeakageError)
    msg = str(exc.value)
    assert "[5]" in msg and f"split={split}" in msg

    # Dataset도 생성되지 않는다 (필터한 split과 상관없이)
    with pytest.raises(ManifestValidationError):
        ManifestDataset(manifest_path, data_root, split="test")


# v3-8: real·gray=True는 test에만, fake·gray=True는 train에 있으면 통과
def test_gray_real_test_only_and_gray_fake_train_passes(build, capsys):
    rows = gray_rows()
    for r in rows:
        if r["origin_id"] == "f0":  # 가짜, train
            r["gray"] = "True"
    manifest_path, data_root = build(rows, columns=GRAY_COLUMNS)
    df = mf.load_manifest(manifest_path, data_root)
    mf.validate(df, data_root)
    out = capsys.readouterr().out
    assert "train/val 흑백 실제 0개" in out
    assert "흑백 행: 실제 test 2장, 가짜 train 2장 / val 0장 / test 0장" in out


# v3-9: summary의 label × split × gray 표에서 (real, test, True) 칸 == df로 직접 센 장수
def test_summary_gray_table_matches_direct_count(build, capsys):
    rows = gray_rows()
    for r in rows:
        if r["origin_id"] == "k0" or r["origin_id"] == "f1":
            r["gray"] = "True"
    manifest_path, data_root = build(rows, columns=GRAY_COLUMNS)
    df = mf.load_manifest(manifest_path, data_root)
    capsys.readouterr()

    returned = mf.summary(df)
    out = capsys.readouterr().out
    assert "[장수표] label × split × gray" in out

    table = mf.gray_count_table(df)
    direct = int(((df["label"] == "real") & (df["split"] == "test") & df["gray"]).sum())
    assert direct == 3  # r3 두 행 + k0
    assert table.loc[("real", "test"), True] == direct
    for (label, split), row in table.iterrows():
        sel = (df["label"] == label) & (df["split"] == split)
        assert row[True] == int((sel & df["gray"]).sum())
        assert row[False] == int((sel & ~df["gray"]).sum())
    # summary 반환값은 기존 표(label × condition × split) 그대로
    assert list(returned.index.names) == ["label", "condition"]


def test_gray_table_has_both_columns_when_all_false(build):
    manifest_path, data_root = build(valid_rows())
    df = mf.load_manifest(manifest_path, data_root)
    table = mf.gray_count_table(df)
    assert list(table.columns) == [False, True]
    assert (table[True] == 0).all()


# dataset.py는 바꾸지 않는다: filter가 gray 열을 그대로 넘기는지 확인
def test_dataset_df_keeps_gray_column(build):
    manifest_path, data_root = build(gray_rows(), columns=GRAY_COLUMNS)
    ds = ManifestDataset(manifest_path, data_root, split="test")
    assert ds.df["gray"].dtype == bool
    assert ds.df.loc[ds.df["origin_id"] == "r3", "gray"].all()
    assert not ds.df.loc[ds.df["origin_id"] != "r3", "gray"].any()
