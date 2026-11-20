"""조건별 평가 (SPEC-03).

실행: python -m src.model.evaluate --config <평가 설정 JSON 경로>

torch가 필요해 Colab에서만 실행한다. torch가 필요 없는 로직은 eval_config.py, eval_report.py에 있고
로컬 pytest로 확인한다 (완료 기준 A).

출력 (run_dir/eval/{eval_name}/)
- eval_config.json: 평가 설정 + 체크포인트 sha256·epoch + manifest_sha256 + 학습 때 manifest_sha256
                    + git 커밋 해시 + torch·torchvision 버전 + GPU 이름
- predictions.csv, summary.csv, drop.csv (eval_report.py 형식)
- eval_log.txt: 화면 출력 전체
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from . import eval_report as er
from .dataset import ManifestDataset
from .eval_config import (
    check_manifest_version,
    create_output_dir,
    ensure_output_dir_absent,
    eval_output_dir,
    load_eval_config,
)
from .resume import MANIFEST_KEY
from .train import IMAGENET_MEAN, IMAGENET_STD, build_model, env_info, torch_load
from .transforms import CropWrapper

# run_dir/config.json(학습 기록)에서 읽는 키 (입력 1: 모델 이름과 crop_size는 평가 설정에 쓰지 않는다)
TRAIN_RECORD_KEYS = ("model", "pretrained", "crop_size", "seed", MANIFEST_KEY)


# ---------------------------------------------------------------- 로그


class _LogSink:
    """로그가 쌓이는 곳. stdout·stderr의 _Tee가 함께 쓴다.

    출력 폴더를 만들기 전(manifest 검사·버전 확인 중)에는 메모리에 모아 두고,
    attach(file)를 부르면 모아 둔 내용을 파일에 쓰고 그 뒤로는 파일에 바로 쓴다.
    쓸 때마다 flush한다 (DataLoader 워커 fork 시 로그 중복 방지, train.py _Tee와 같은 이유).
    """

    def __init__(self):
        self.target = io.StringIO()

    def attach(self, log_file) -> None:
        log_file.write(self.target.getvalue())
        log_file.flush()
        self.target = log_file

    def detach(self) -> None:
        self.target = io.StringIO()

    def write(self, text) -> None:
        self.target.write(text)
        self.target.flush()


class _Tee:
    """화면(stream)과 _LogSink에 함께 쓴다."""

    def __init__(self, stream, sink: _LogSink):
        self.stream = stream
        self.sink = sink

    def write(self, text):
        self.stream.write(text)
        self.sink.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return False


# ---------------------------------------------------------------- 준비


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_record(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"학습 기록이 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    missing = [k for k in TRAIN_RECORD_KEYS if k not in record]
    if missing:
        raise KeyError(f"학습 기록 {path}에 필요한 키가 없습니다: {', '.join(missing)}")
    return record


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(title)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
        print(table.to_string(index=False) if len(table) else "(없음)")


# ---------------------------------------------------------------- 추론


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """전체 행을 한 번 추론한다. 반환: (로짓 float64, row_index), 데이터 순서 그대로."""
    model.eval()
    all_logits, all_rows = [], []
    for images, _, row_index in loader:
        logits = model(images.to(device)).squeeze(1)
        all_logits.append(logits.cpu())
        all_rows.append(row_index)
    # train.py evaluate와 같게: float32 로짓을 CPU로 옮긴 뒤 float64로
    logits = torch.cat(all_logits).double().numpy()
    rows = torch.cat(all_rows).numpy()
    return logits, rows


# ---------------------------------------------------------------- main


def run(config: dict, sink: _LogSink | None = None) -> Path:
    run_dir = Path(config["run_dir"])

    # 1. 평가 설정(이미 읽음)과 학습 기록
    record = load_train_record(run_dir)

    # 2. 출력 폴더가 이미 있으면 멈춘다 (만드는 것은 검사가 끝난 뒤)
    out_dir = eval_output_dir(config)
    ensure_output_dir_absent(out_dir)

    # 3. ManifestDataset을 한 번만 만든다 (split 지정, condition은 거르지 않음. manifest 검사도 여기서 한 번)
    base = ManifestDataset(
        config["manifest_path"],
        config["data_root"],
        source=config["source"],
        condition=None,
        split=config["split"],
        transform=None,
    )
    rows = base.df
    if len(rows) == 0:
        raise ValueError(f"평가할 행이 없습니다 (split={config['split']}, source={config['source']})")

    # 4. manifest 버전 확인 (처리 방식 5)
    version_msg = check_manifest_version(
        rows,
        config["split"],
        base.manifest_sha256,
        record[MANIFEST_KEY],
        config["train_manifest_path"],
    )

    # 검사 통과 → 출력 폴더를 만들고 지금까지의 로그를 eval_log.txt로 옮긴다
    create_output_dir(out_dir)
    with open(out_dir / "eval_log.txt", "w", encoding="utf-8") as log_file:
        if sink is not None:
            sink.attach(log_file)
        try:
            _evaluate(config, record, base, rows, version_msg, out_dir)
        except BaseException:
            # 에러 traceback을 eval_log.txt에 남긴다.
            # 화면에는 raise 뒤 파이썬이 출력한다 (여기서 화면에도 쓰면 두 번 나온다)
            traceback.print_exc(file=log_file)
            raise
        finally:
            if sink is not None:
                sink.detach()  # 파일을 닫기 전에 떼어 둔다
    return out_dir


def _evaluate(config, record, base, rows, version_msg, out_dir: Path) -> None:
    run_dir = Path(config["run_dir"])
    ckpt_path = run_dir / f"{config['checkpoint']}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")
    ckpt_sha256 = file_sha256(ckpt_path)
    ckpt = torch_load(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = env_info()

    # eval_config.json (출력 2)
    eval_record = {
        **config,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "checkpoint_epoch": ckpt["epoch"],
        "manifest_sha256": base.manifest_sha256,
        "train_manifest_sha256": record[MANIFEST_KEY],
        "train_model": record["model"],
        "train_crop_size": record["crop_size"],
        **env,
    }
    with open(out_dir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(eval_record, f, ensure_ascii=False, indent=2)

    # 5. 시작 로그
    print("=" * 60)
    print("평가 시작:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("평가 설정:", json.dumps(config, ensure_ascii=False))
    print(f"학습 기록: model={record['model']}, crop_size={record['crop_size']}")
    print(f"체크포인트: {ckpt_path} (epoch {ckpt['epoch']}, sha256={ckpt_sha256})")
    print("manifest_sha256:", base.manifest_sha256)
    print(version_msg)
    counts = rows.groupby(["source", "condition", "label"]).size().rename("행 수").reset_index()
    _print_table("[행 수] (source, condition, label)", counts)
    _print_table("[파일 확장자 분포] (source, condition, label)", er.extension_table(rows).reset_index())
    for warning in er.format_bias_warnings(rows):
        print(f"[경고] {warning}")
    print("장치:", device, f"({env['gpu_name']})" if env["gpu_name"] else "")
    if device.type != "cuda":
        print("경고: GPU 없음, CPU로 평가합니다 (Colab 런타임 유형을 확인하세요)")
    print("실행 환경:", json.dumps(env, ensure_ascii=False))

    # 6. 중앙 크롭 → ToTensor → Normalize (SPEC-02 1절과 같다)
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dataset = CropWrapper(base, record["crop_size"], "eval", record["seed"], to_tensor)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        persistent_workers=False,
    )

    # 가중치는 바로 체크포인트로 덮어쓰므로 ImageNet 가중치를 내려받지 않는다 (구조는 같다)
    model = build_model({**record, "pretrained": False})
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # 7. 전체 행 추론 → predictions.csv
    start = time.time()
    logits, row_index = predict_logits(model, loader, device)
    if not np.array_equal(row_index, rows.index.to_numpy()):
        raise RuntimeError("추론 결과의 행 순서가 manifest 행 순서와 다릅니다")
    predictions = er.build_predictions(rows, logits)
    er.write_predictions(out_dir / "predictions.csv", predictions)
    print(f"추론 {len(predictions)}장, {time.time() - start:.1f}s → predictions.csv")

    # 8. summary.csv, drop.csv
    summary = er.summarize(predictions)
    drop = er.compute_drop(predictions)
    er.write_table(out_dir / "summary.csv", summary)
    er.write_table(out_dir / "drop.csv", drop)

    _print_table("[summary] 조건별 지표", summary[summary["group"] == "condition"])
    _print_table("[summary] 생성기별 미탐률", summary[summary["group"] == "generator"])
    _print_table("[drop] 원본 대비 변화 (%p)", drop)
    for mode in dict.fromkeys(drop["baseline"]):
        print(f"drop 기준 {mode}: {er.BASELINE_DESCRIPTIONS[mode]}")
    print("결과 폴더:", out_dir)
    print("평가 끝:", time.strftime("%Y-%m-%d %H:%M:%S"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="SPEC-03 조건별 평가")
    parser.add_argument("--config", required=True, help="평가 설정 JSON 경로 (src/model/configs/*_eval_*.json)")
    args = parser.parse_args(argv)

    config = load_eval_config(args.config)

    # 출력 폴더를 만들기 전의 에러(설정·manifest 검사·버전 확인)는 화면에만 나온다 (폴더를 남기지 않기 위함)
    sink = _LogSink()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_stdout, sink)
    sys.stderr = _Tee(old_stderr, sink)
    try:
        run(config, sink)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


if __name__ == "__main__":
    main()
