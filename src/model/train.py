"""베이스라인 학습 (SPEC-02).

실행: python -m src.model.train --config <설정 JSON 경로>

torch는 이 파일에서만 import한다 (로컬 Windows에서는 torch import가 막혀 있어 Colab에서만 실행).
torch가 필요 없는 로직은 config.py, transforms.py(CropWrapper), run_log.py, resume.py, metrics.py, select.py에 있고
로컬 pytest로 확인한다.

출력 (output_dir/run_name/)
- config.json: 설정 + manifest_sha256 + git 커밋 해시 + torch·torchvision 버전 + GPU 이름 (처음 실행에서만 저장)
- metrics.csv: epoch별 지표 (run_log.py 형식)
- last.pt: 모델, 옵티마이저, epoch, 난수 상태 (매 epoch 끝)
- best.pt: val_loss 최소 epoch의 모델, epoch, val_loss
- train_log.txt: 화면 출력 전체 (재개 시 이어쓰기)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights, resnet50

from . import resume as rs
from . import run_log
from . import select as sel
from .config import TRAIN_GENERATORS_KEY, load_config
from .dataset import LABEL_TO_INDEX, ManifestDataset
from .metrics import average_precision, label_metrics, roc_auc
from .transforms import CropWrapper

# ImageNet 정규화 값 (SPEC-02 1절). ResNet50_Weights.IMAGENET1K_V1.transforms()는
# 리사이즈·중앙 크롭이 들어 있어 쓰지 않는다
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- 로그


class _Tee:
    """화면 출력과 train_log.txt에 함께 쓴다.

    쓸 때마다 파일을 flush한다. DataLoader 워커가 fork될 때 버퍼에 남은 내용이
    워커 종료 시 다시 flush되어 로그가 중복되는 것을 막기 위함이다.
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return False


# ---------------------------------------------------------------- 재현성


def seed_everything(seed: int) -> None:
    """Python random, NumPy, torch(CPU·CUDA) 난수를 seed로 고정한다 (SPEC-02 4절)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CUDA 장치 전부 포함


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 워커의 Python·NumPy 난수를 고정한다.

    torch.initial_seed()는 워커마다 (DataLoader generator가 정한 base_seed + worker_id)다.
    generator를 seed로 고정하므로 워커 seed도 고정된다.
    (학습 크롭 좌표는 이 값과 상관없이 (seed, epoch, row_index)로 정해진다)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def epoch_generator_seed(seed: int, epoch: int) -> int:
    """epoch별 DataLoader 셔플 generator seed.

    epoch마다 (seed, epoch)로 generator를 다시 고정해, 재개한 실행도 끊기지 않은 실행과
    같은 epoch에서 같은 셔플 순서를 쓰게 한다.
    """
    return int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------- 실행 환경


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def env_info() -> dict:
    """resume.ENV_KEYS 키 이름 그대로의 실행 환경 값."""
    info = {
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    assert tuple(info) == rs.ENV_KEYS
    return info


# ---------------------------------------------------------------- 저장


def atomic_torch_save(obj, path: Path) -> None:
    """임시 파일에 저장한 뒤 os.replace로 이름을 바꾼다 (저장 도중 끊겨도 이전 파일이 깨지지 않음)."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def torch_load(path: Path):
    # 난수 상태(Python 튜플, NumPy 배열)가 들어 있어 weights_only=False로 읽는다 (직접 만든 파일만 읽음)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- 모델·데이터


def build_model(config: dict) -> nn.Module:
    if config["model"] != "resnet50":
        raise ValueError(f"SPEC-02는 model='resnet50'만 지원합니다: {config['model']!r}")
    weights = ResNet50_Weights.IMAGENET1K_V1 if config["pretrained"] else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)  # 출력 1개 = fake 로짓
    return model


def build_datasets(config: dict):
    """split별 CropWrapper와 고르기 전 df를 돌려준다.

    ManifestDataset을 지금처럼 만든 뒤(전체 manifest 검사 포함) dataset.df를
    select_train_rows 결과로 바꾼다 (보완 4). train_generators 키가 없으면 df는 그대로다.
    """
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_generators = config.get(TRAIN_GENERATORS_KEY)
    datasets, before = {}, {}
    for split, mode in (("train", "train"), ("val", "eval")):
        base = ManifestDataset(
            config["manifest_path"],
            config["data_root"],
            source=config["source"],
            condition="original",
            split=split,
            transform=None,
        )
        before[split] = base.df
        base.df = sel.select_train_rows(base.df, train_generators, split)
        datasets[split] = CropWrapper(base, config["crop_size"], mode, config["seed"], to_tensor)
    return datasets, before


def build_loader(dataset, config: dict, shuffle: bool, generator=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        persistent_workers=False,  # True면 워커가 첫 epoch 값을 계속 써서 크롭 위치가 안 바뀐다
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


# ---------------------------------------------------------------- 학습·평가


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """한 epoch 학습. 반환: 샘플 평균 train_loss."""
    model.train()
    total, count = 0.0, 0
    for images, labels, _ in loader:
        images = images.to(device)
        targets = labels.to(device).float()
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """val 전체로 val_loss와 지표 5개를 계산한다."""
    model.eval()
    total, count = 0.0, 0
    all_logits, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(device)
        targets = labels.to(device).float()
        logits = model(images).squeeze(1)
        total += criterion(logits, targets).item() * len(labels)
        count += len(labels)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits).double()
    y_true = torch.cat(all_labels).numpy()
    y_pred = (logits >= 0).long().numpy()  # 로짓 0 = 점수 0.5. 같으면 fake (AP 임계값 규칙과 같게)
    scores = torch.sigmoid(logits).numpy()  # fake일 확률

    out = {"val_loss": total / count}
    out.update({f"val_{k}": v for k, v in label_metrics(y_true, y_pred).items()})
    out["val_ap"] = average_precision(y_true, scores)
    out["val_auc"] = roc_auc(y_true, scores)
    return out


def _fmt(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


# ---------------------------------------------------------------- 재개


def prepare_run(run_dir: Path, record: dict, model, optimizer) -> int:
    """처음 실행이면 config.json을 쓰고 0을, 재개면 상태를 불러오고 last epoch를 돌려준다."""
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    if not last_path.exists():
        # 처음 실행 (또는 epoch 1의 last.pt 저장 전에 끊긴 실행)
        if best_path.exists():
            raise rs.ResumeError(f"last.pt 없이 best.pt만 있습니다: {best_path}")
        if metrics_path.exists():
            # 자동 삭제는 "epoch 1 행 하나만 있음"일 때만 한다 (epoch 1의 metrics.csv 추가 후 last.pt 저장 전에 끊긴 경우).
            # 그 외에는 Drive 연결 문제로 last.pt가 안 보이는 경우일 수 있어 기록을 지우지 않고 멈춘다
            rows = run_log.read_rows(metrics_path)
            if [r["epoch"] for r in rows] == [1]:
                print(f"[처음 실행] last.pt가 없어 끝나지 않은 epoch 1의 metrics.csv를 지웁니다: {metrics_path}")
                metrics_path.unlink()
            else:
                raise rs.ResumeError(
                    f"last.pt 없이 metrics.csv가 있습니다 (행 {len(rows)}개). "
                    f"Drive 연결 상태와 output_dir를 확인하세요: {run_dir}"
                )
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[처음 실행] config.json 저장: {config_path}")
        return 0

    # 재개
    if not config_path.exists():
        raise rs.ResumeError(f"last.pt는 있는데 config.json이 없습니다: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        saved = json.load(f)
    for warning in rs.check_resume_config(saved, record):
        print(f"[경고] {warning}")

    last = torch_load(last_path)
    last_epoch = last["epoch"]
    rows = run_log.read_rows(metrics_path)
    best_epoch = torch_load(best_path)["epoch"] if best_path.exists() else None
    plan = rs.plan_resume(last_epoch, run_log.loss_rows(rows), best_epoch)
    run_log.rewrite_kept(metrics_path, rows, plan.kept_rows)
    print(f"[재개] last.pt epoch={last_epoch}, metrics.csv 행 {len(plan.kept_rows)}개 유지, best epoch={plan.best_epoch}")

    if plan.best_action == rs.REBUILD_BEST_FROM_LAST:
        val_loss = dict(plan.kept_rows)[last_epoch]
        atomic_torch_save({"model": last["model"], "epoch": last_epoch, "val_loss": val_loss}, best_path)
        print(f"[재개] best.pt를 last.pt 모델로 다시 만들었습니다 (epoch {last_epoch}, val_loss {val_loss:.6f})")

    model.load_state_dict(last["model"])
    optimizer.load_state_dict(last["optimizer"])
    set_rng_state(last["rng"])
    return last_epoch


# ---------------------------------------------------------------- main


def run(config: dict) -> None:
    run_dir = Path(config["output_dir"]) / config["run_name"]
    metrics_path = run_dir / "metrics.csv"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, before = build_datasets(config)
    manifest_sha256 = datasets["train"].dataset.manifest_sha256
    env = env_info()
    record = {**config, rs.MANIFEST_KEY: manifest_sha256, **env}

    # 학습 시작 로그 (SPEC-02 6절)
    print("=" * 60)
    print("학습 시작:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("config:", json.dumps(config, ensure_ascii=False))
    print("manifest_sha256:", manifest_sha256)
    # 보완 4: 학습 생성기, 고르기 전 표, (기존 형식) 고른 뒤 label 행 수, 고른 뒤 gray 표
    train_generators = config.get(TRAIN_GENERATORS_KEY)
    print(sel.train_generators_line(train_generators))
    for line in sel.generator_table_lines(before, train_generators):
        print(line)
    index_to_label = {v: k for k, v in LABEL_TO_INDEX.items()}
    for split, ds in datasets.items():
        counts = ds.dataset.df["label"].value_counts()
        by_label = ", ".join(f"{name} {int(counts.get(name, 0))}" for name in index_to_label.values())
        print(f"split {split} (condition=original): {by_label}, 합계 {len(ds)}")
    for line in sel.gray_table_lines({split: ds.dataset.df for split, ds in datasets.items()}):
        print(line)
    print("장치:", device, f"({env['gpu_name']})" if env["gpu_name"] else "")
    if device.type != "cuda":
        print("!" * 60)
        print("경고: GPU 없음, CPU로 학습합니다 (Colab 런타임 유형을 확인하세요)")
        print("!" * 60)
    print("실행 환경:", json.dumps(env, ensure_ascii=False))

    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    last_epoch = prepare_run(run_dir, record, model, optimizer)
    if last_epoch >= config["epochs"]:
        print(f"이미 완료: last.pt epoch={last_epoch} ≥ epochs={config['epochs']}")
        return

    generator = torch.Generator()
    train_loader = build_loader(datasets["train"], config, shuffle=True, generator=generator)
    val_loader = build_loader(datasets["val"], config, shuffle=False)

    for epoch in range(last_epoch + 1, config["epochs"] + 1):
        start = time.time()
        datasets["train"].set_epoch(epoch)
        generator.manual_seed(epoch_generator_seed(config["seed"], epoch))

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, **val, "epoch_seconds": time.time() - start}

        # 저장 순서 (SPEC-02 5절): metrics.csv → last.pt → (최소이면) best.pt
        run_log.append_row(metrics_path, row)
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "rng": rng_state(),
            },
            last_path,
        )
        new_best = rs.is_new_best(run_log.loss_rows(run_log.read_rows(metrics_path)), epoch)
        if new_best:
            atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "val_loss": val["val_loss"]}, best_path)

        print(
            f"[epoch {epoch}/{config['epochs']}] "
            + ", ".join(f"{c}={_fmt(row[c])}" for c in run_log.COLUMNS if c != "epoch")
            + (" (best 저장)" if new_best else "")
        )

    print("학습 끝:", time.strftime("%Y-%m-%d %H:%M:%S"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="SPEC-02 베이스라인 학습")
    parser.add_argument("--config", required=True, help="설정 JSON 경로 (src/model/configs/)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_dir = Path(config["output_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "train_log.txt", "a", encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_stdout, log_file)
        sys.stderr = _Tee(old_stderr, log_file)
        try:
            run(config)
        except BaseException:
            # 에러 traceback을 train_log.txt에 남긴다.
            # 화면에는 아래 raise 뒤 파이썬이 그대로 출력한다 (여기서 화면에도 쓰면 화면에 두 번 나온다)
            traceback.print_exc(file=log_file)
            raise
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


if __name__ == "__main__":
    main()
