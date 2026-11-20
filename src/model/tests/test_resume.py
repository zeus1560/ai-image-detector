"""SPEC-02 A: resume.py 테스트.

완료 기준 3가지(정상 / last 후 best 전 끊김 복구 / 불일치 에러)와
추가 요청 2가지(best.pt 없음 복구 / metrics 행 누락·중복 에러)를 확인한다.
"""

import pytest

from src.model.resume import (
    KEEP_BEST,
    REBUILD_BEST_FROM_LAST,
    ResumeError,
    best_epoch_of,
    plan_resume,
)


# ---- 완료 기준 3가지 ----

def test_normal_keep():
    """정상: best.pt epoch가 val_loss 최소 epoch와 같으면 그대로."""
    rows = [(1, 0.50), (2, 0.40), (3, 0.45)]
    plan = plan_resume(3, rows, best_epoch=2)
    assert plan.best_action == KEEP_BEST
    assert plan.best_epoch == 2
    assert plan.kept_rows == rows


def test_normal_with_trailing_row_trimmed():
    """정상(metrics 추가 후 last.pt 전 끊김): last보다 뒤쪽 행은 지운다."""
    rows = [(1, 0.50), (2, 0.40), (3, 0.45), (4, 0.10)]
    plan = plan_resume(3, rows, best_epoch=2)
    assert plan.kept_rows == rows[:3]
    assert plan.best_action == KEEP_BEST  # 지워진 epoch 4의 낮은 loss는 반영하지 않는다


def test_crash_after_last_before_best_rebuilds():
    """last.pt 후 best.pt 전 끊김: 최소 epoch == last epoch → last.pt로 best.pt 다시 만들기."""
    rows = [(1, 0.50), (2, 0.40), (3, 0.30)]
    plan = plan_resume(3, rows, best_epoch=2)
    assert plan.best_action == REBUILD_BEST_FROM_LAST
    assert plan.best_epoch == 3


def test_mismatch_raises():
    """불일치: best.pt epoch ≠ 최소 epoch이고 최소 epoch ≠ last epoch → 에러."""
    rows = [(1, 0.50), (2, 0.40), (3, 0.45)]
    with pytest.raises(ResumeError, match="best.pt 불일치"):
        plan_resume(3, rows, best_epoch=1)


# ---- 추가: best.pt 없음 ----

def test_no_best_file_rebuilds_when_min_is_last():
    """best.pt 없음 → 복구: 1 epoch에서 last.pt 후 best.pt 전에 끊긴 경우."""
    plan = plan_resume(1, [(1, 0.50)], best_epoch=None)
    assert plan.best_action == REBUILD_BEST_FROM_LAST
    assert plan.best_epoch == 1


def test_no_best_file_raises_when_min_is_not_last():
    """best.pt가 없는데 최소 epoch가 last가 아니면 복구할 수 없다 → 에러."""
    with pytest.raises(ResumeError):
        plan_resume(2, [(1, 0.30), (2, 0.40)], best_epoch=None)


# ---- 추가: metrics 행 누락·중복 ----

def test_missing_row_raises():
    with pytest.raises(ResumeError, match=r"누락 \[2\]"):
        plan_resume(3, [(1, 0.5), (3, 0.4)], best_epoch=3)


def test_duplicate_row_raises():
    with pytest.raises(ResumeError, match=r"중복 \[2\]"):
        plan_resume(3, [(1, 0.5), (2, 0.4), (2, 0.4), (3, 0.45)], best_epoch=2)


def test_duplicate_after_last_is_trimmed_not_error():
    """last보다 뒤쪽 행은 지우는 대상이라 중복이어도 에러가 아니다."""
    plan = plan_resume(2, [(1, 0.5), (2, 0.4), (3, 0.3), (3, 0.3)], best_epoch=2)
    assert plan.kept_rows == [(1, 0.5), (2, 0.4)]


# ---- 동점 규칙과 입력 검사 ----

def test_tie_prefers_earlier_epoch():
    assert best_epoch_of([(1, 0.4), (2, 0.4)]) == 1
    assert plan_resume(2, [(1, 0.4), (2, 0.4)], best_epoch=1).best_action == KEEP_BEST


@pytest.mark.parametrize("last", [0, -1, None, 1.0])
def test_invalid_last_epoch_raises(last):
    with pytest.raises(ResumeError):
        plan_resume(last, [(1, 0.5)], best_epoch=1)


def test_nan_val_loss_raises():
    with pytest.raises(ResumeError):
        plan_resume(2, [(1, 0.5), (2, float("nan"))], best_epoch=1)


# ---- is_new_best (확정 후 보완 2-3) ----

from src.model.resume import check_resume_config, is_new_best  # noqa: E402


def test_is_new_best_first_epoch_true():
    assert is_new_best([(1, 0.7)], 1) is True


def test_is_new_best_tie_false():
    """동점이면 앞 epoch를 유지하므로 거짓."""
    assert is_new_best([(1, 0.5), (2, 0.5)], 2) is False


def test_is_new_best_lower_true_higher_false():
    assert is_new_best([(1, 0.5), (2, 0.4)], 2) is True
    assert is_new_best([(1, 0.5), (2, 0.6)], 2) is False


def test_is_new_best_missing_epoch_raises():
    with pytest.raises(ResumeError):
        is_new_best([(1, 0.5)], 2)


# ---- check_resume_config ----

def saved_config():
    return {
        "run_name": "cifake_baseline",
        "lr": 0.0001,
        "crop_size": None,
        "epochs": 5,
        "manifest_sha256": "aaa",
        "git_commit": "7d39f41",
        "torch_version": "2.5.0",
        "torchvision_version": "0.20.0",
        "gpu_name": "Tesla T4",
    }


def test_resume_config_same_no_warning():
    assert check_resume_config(saved_config(), saved_config()) == []


def test_resume_config_setting_differs_raises():
    current = {**saved_config(), "lr": 0.001}
    with pytest.raises(ResumeError, match="lr"):
        check_resume_config(saved_config(), current)


def test_resume_config_missing_key_raises():
    current = saved_config()
    del current["epochs"]
    with pytest.raises(ResumeError, match="epochs"):
        check_resume_config(saved_config(), current)


def test_resume_config_manifest_differs_raises():
    current = {**saved_config(), "manifest_sha256": "bbb"}
    with pytest.raises(ResumeError, match="manifest_sha256"):
        check_resume_config(saved_config(), current)


def test_resume_config_commit_only_warns():
    current = {**saved_config(), "git_commit": "37a5390"}
    warnings = check_resume_config(saved_config(), current)
    assert len(warnings) == 1
    assert "git_commit" in warnings[0]


def test_resume_config_env_keys_warn():
    current = {**saved_config(), "torch_version": "2.6.0", "gpu_name": "NVIDIA L4"}
    warnings = check_resume_config(saved_config(), current)
    assert len(warnings) == 2
