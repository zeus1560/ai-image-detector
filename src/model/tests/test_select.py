"""SPEC-02 확정 후 보완 4: select.py(학습 생성기 행 고르기)와 train_generators 설정 테스트.

train.py와 같은 경로로 확인한다: ManifestDataset(source, original, split)을 만든 뒤 df를 고른다.
"""

import json
import warnings
from pathlib import Path

import pytest

from conftest import make_row
from src.model import resume as rs
from src.model import select as sel
from src.model.config import TRAIN_GENERATORS_KEY, ConfigError, load_config
from src.model.dataset import ManifestDataset
from src.model.manifest import REQUIRED_COLUMNS

GRAY_COLUMNS = REQUIRED_COLUMNS + ["gray"]
GENERATORS = ("sd14", "sd15", "wukong")
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def row(origin_id, label, split, generator, condition="original", gray="False"):
    r = make_row(origin_id, label, split, source="genimage", condition=condition)
    r["generator"] = generator
    r["path"] = f"images/genimage/{condition}/{generator}/{origin_id}_{condition}.png"
    r["gray"] = gray
    return r


def three_generator_rows():
    """공유 실제 + 생성기 3개. split마다 real 3·2·2, 생성기별 fake 2·1·1. jpeg75 변형본과 흑백 가짜 포함."""
    rows = []
    reals = {"train": ["r0", "r1", "r2"], "val": ["r3", "r4"], "test": ["r5", "r6"]}
    for split, oids in reals.items():
        for oid in oids:
            rows.append(row(oid, "real", split, "real"))
    fakes = {"train": 2, "val": 1, "test": 1}
    for g in GENERATORS:
        for split, n in fakes.items():
            for i in range(n):
                # sd15 train의 첫 가짜는 흑백 (gray 표 확인용)
                gray = "True" if (g == "sd15" and split == "train" and i == 0) else "False"
                rows.append(row(f"{g}_{split}{i}", "fake", split, g, gray=gray))
    # 변형본: original로 거르면 빠져야 한다
    rows += [dict(r, condition="jpeg75", path=r["path"].replace("original", "jpeg75")) for r in list(rows)]
    return rows


def split_df(build, rows, split):
    manifest_path, data_root = build(rows, columns=GRAY_COLUMNS)
    ds = ManifestDataset(manifest_path, data_root, source="genimage", condition="original", split=split)
    return ds.df


# ---------------------------------------------------------------- A1: 키 없음


def test_none_returns_df_unchanged(build):
    df = split_df(build, three_generator_rows(), "train")
    assert sel.select_train_rows(df, None, "train") is df


# ---------------------------------------------------------------- A3: ["sd14"]만 고르기


@pytest.mark.parametrize("split,n_real,n_fake", [("train", 3, 2), ("val", 2, 1)])
def test_select_sd14_keeps_all_real_and_only_sd14_fake(build, split, n_real, n_fake):
    df = split_df(build, three_generator_rows(), split)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # real:fake 장수 차이 경고는 따로 확인
        out = sel.select_train_rows(df, ["sd14"], split)
    real = out[out["label"] == "real"]
    fake = out[out["label"] == "fake"]
    assert len(real) == n_real == int((df["label"] == "real").sum())
    assert set(fake["generator"]) == {"sd14"}
    assert len(fake) == n_fake
    assert set(out["condition"]) == {"original"}
    # 원래 행 인덱스 유지 (CropWrapper가 df.loc[row_index]로 path를 찾음)
    assert out.index.isin(df.index).all()
    assert (out["path"] == df.loc[out.index, "path"]).all()


def test_imbalance_warns_and_continues(build):
    df = split_df(build, three_generator_rows(), "train")
    with pytest.warns(UserWarning, match="real 3 : fake 4"):
        out = sel.select_train_rows(df, ["sd14", "sd15"], "train")
    assert len(out) == 7


def test_balanced_selection_no_warning(build):
    rows = [r for r in three_generator_rows() if not (r["label"] == "real" and r["origin_id"] == "r2")]
    df = split_df(build, rows, "train")  # real 2 : sd14 fake 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sel.select_train_rows(df, ["sd14"], "train")


# ---------------------------------------------------------------- A4: manifest에 없는 생성기


def test_unknown_generator_named_in_error(build):
    df = split_df(build, three_generator_rows(), "val")
    with pytest.raises(sel.SelectionError) as exc:
        sel.select_train_rows(df, ["sd14", "dalle3"], "val")
    msg = str(exc.value)
    assert "[a]" in msg and "dalle3" in msg and "split=val" in msg
    assert "'sd14'" not in msg.split("[a]")[1]  # 있는 생성기는 목록에 넣지 않는다


# ---------------------------------------------------------------- A5: 같은 실제 origin_id가 두 행


def test_duplicate_real_origin_id_with_other_generator_is_error(build):
    rows = three_generator_rows()
    # 공유 실제 r0이 sd15 폴더에 한 번 더 등록된 경우 (generator만 다름)
    rows.append(row("r0", "real", "train", "sd15"))
    df = split_df(build, rows, "train")
    with pytest.raises(sel.SelectionError) as exc:
        sel.select_train_rows(df, ["sd14"], "train")
    msg = str(exc.value)
    assert "[c]" in msg and "origin_id" in msg and "r0" in msg and "split=train" in msg


def test_duplicate_real_origin_id_both_generator_real_is_error(build):
    rows = three_generator_rows()
    dup = row("r1", "real", "train", "real")
    dup["path"] = "images/genimage/original/real/r1_copy.png"
    rows.append(dup)
    df = split_df(build, rows, "train")
    with pytest.raises(sel.SelectionError) as exc:
        sel.select_train_rows(df, ["sd14"], "train")
    msg = str(exc.value)
    assert "[c]" in msg and "r1" in msg
    assert "[b]" not in msg


def test_duplicate_path_is_error(build):
    rows = three_generator_rows()
    dup = row("r9", "real", "train", "real")
    dup["path"] = rows[0]["path"]  # r0과 같은 파일
    rows.append(dup)
    df = split_df(build, rows, "train")
    with pytest.raises(sel.SelectionError) as exc:
        sel.select_train_rows(df, ["sd14"], "train")
    msg = str(exc.value)
    assert "[c]" in msg and "path" in msg and rows[0]["path"] in msg


# ---------------------------------------------------------------- A6: 실제 행의 generator가 real 아님


def test_real_row_with_non_real_generator_is_error(build):
    rows = three_generator_rows()
    for r in rows:
        if r["origin_id"] == "r3":
            r["generator"] = "sd14"
    df = split_df(build, rows, "val")
    with pytest.raises(sel.SelectionError) as exc:
        sel.select_train_rows(df, ["sd14"], "val")
    msg = str(exc.value)
    assert "[b]" in msg and "split=val" in msg and "sd14" in msg


# ---------------------------------------------------------------- A2: 설정 형식 검사


def base_config():
    with open(CONFIGS_DIR / "cifake_baseline.json", encoding="utf-8") as f:
        return json.load(f)


def write(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "value,reason",
    [
        ([], "빈 목록"),
        (["sd14", "sd14"], "중복"),
        (["sd14", "real"], "'real'"),
        (["sd14", 3], "문자열이 아닌 값"),
        (["sd14", None], "문자열이 아닌 값"),
        ([""], "빈 문자열"),
        ("sd14", "목록이어야"),
        (None, "목록이어야"),
    ],
)
def test_bad_train_generators_raise_config_error_with_reason(tmp_path, value, reason, capsys):
    data = base_config()
    data[TRAIN_GENERATORS_KEY] = value
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, data))
    assert reason in str(exc.value)
    assert TRAIN_GENERATORS_KEY in str(exc.value)


def test_valid_train_generators_loaded(tmp_path):
    data = base_config()
    data[TRAIN_GENERATORS_KEY] = ["sd14", "sd15"]
    assert load_config(write(tmp_path, data))[TRAIN_GENERATORS_KEY] == ["sd14", "sd15"]


# ---------------------------------------------------------------- A1·A7: 키 없는 설정은 지금과 같음


def test_config_without_key_is_unchanged():
    config = load_config(CONFIGS_DIR / "cifake_baseline.json")
    assert TRAIN_GENERATORS_KEY not in config  # 기본값을 채우지 않는다
    assert config == base_config()


def test_cifake_config_compares_with_existing_config_json():
    """cifake_baseline.json으로 만든 설정이 기존 config.json과 check_resume_config에서 에러 없이 비교된다.

    기존 config.json = 보완 4 이전 train.py가 쓴 {**설정 파일 내용, manifest_sha256, 실행 환경}.
    """
    sha = "79e176eeb61bb024e6ffbc5163b5c84fa02f23daf74b9e39c2427086bcb3740c"
    env = {"git_commit": "old", "torch_version": "2.x", "torchvision_version": "0.x", "gpu_name": "T4"}
    saved = {**base_config(), rs.MANIFEST_KEY: sha, **env}

    config = load_config(CONFIGS_DIR / "cifake_baseline.json")
    current = {**config, rs.MANIFEST_KEY: sha, **dict(env, git_commit="new")}
    warnings_ = rs.check_resume_config(saved, current)
    assert warnings_ == ["git_commit가 처음 실행과 다릅니다: old → new"]

    # 대조: 키 없을 때 None을 채웠다면 재개가 막혔을 것이다
    with pytest.raises(rs.ResumeError):
        rs.check_resume_config(saved, {**current, TRAIN_GENERATORS_KEY: None})


# ---------------------------------------------------------------- 학습 시작 로그 (보완 4의 3)


def test_train_generators_line():
    assert sel.train_generators_line(None) == "train_generators: 키 없음: 모든 생성기 사용"
    assert sel.train_generators_line(["sd14"]) == "train_generators: ['sd14']"


def test_generator_table_lines_mark_used_and_counts_match(build):
    rows = three_generator_rows()
    before = {s: split_df(build, rows, s) for s in ("train", "val")}
    lines = sel.generator_table_lines(before, ["sd14"])
    body = lines[1:]
    assert len(body) == 2 * 4  # split 2개 × (real + 생성기 3개)
    for line in body:
        split, gen, label, n, mark = line.split()
        df = before[split]
        assert int(n) == int(((df["generator"] == gen) & (df["label"] == label)).sum())
        assert mark == ("사용" if label == "real" or gen == "sd14" else "제외")


def test_generator_table_lines_all_used_without_key(build):
    before = {"train": split_df(build, three_generator_rows(), "train")}
    assert all(line.endswith("사용") for line in sel.generator_table_lines(before, None)[1:])


def test_gray_table_lines_after_selection(build):
    rows = three_generator_rows()
    after = {}
    for s in ("train", "val"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            after[s] = sel.select_train_rows(split_df(build, rows, s), ["sd14", "sd15"], s)
    lines = sel.gray_table_lines(after)
    assert lines[1].split() == ["train", "real", "gray=False", "3,", "gray=True", "0"]
    assert lines[2].split() == ["train", "fake", "gray=False", "3,", "gray=True", "1"]
    assert lines[4].split() == ["val", "fake", "gray=False", "2,", "gray=True", "0"]
