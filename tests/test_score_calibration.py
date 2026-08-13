"""比分/总进球校准回归测试（engine.learning.score_calibration）。

锁定 2026-08-14 批次：模型比分矩阵系统性偏向低比分（3+ 球被低估），
calibrate_* 向经验先验收缩（shrink=0.4）修正。
"""
from __future__ import annotations

import pytest

from engine.learning.score_calibration import (
    calibrate_score_probs,
    calibrate_total_goals,
    empirical_score_prob,
)

PRIOR = {
    "n_matches": 810,
    "score": {
        "1-0": 0.132, "2-1": 0.102, "1-1": 0.101, "0-1": 0.078,
        "2-0": 0.068, "0-0": 0.067, "3-1": 0.040, "3-2": 0.020,
    },
    "total_goals": {
        "0": 0.067, "1": 0.210, "2": 0.223, "3": 0.221,
        "4": 0.154, "5": 0.065, "6": 0.038,
    },
}


def test_calibrate_score_probs_shrinks_toward_prior():
    # 收缩公式（归一化前）: calibrated = (1-shrink)*model + shrink*prior
    # 0-0: 0.6*0.5 + 0.4*0.9 = 0.66；1-0: 0.6*0.5 + 0.4*0.05 = 0.32
    top = [[0, 0, 0.5], [1, 0, 0.5]]
    prior = {"score": {"0-0": 0.9, "1-0": 0.05}}
    out = calibrate_score_probs(top, prior, shrink=0.4)
    assert len(out) == 2
    assert abs(sum(c[2] for c in out) - 1.0) < 1e-9
    # 0-0 先验高（0.9）→ 校准后份额上升（0.5 → 0.66/0.98 ≈ 0.6735）
    p00 = next(c[2] for c in out if (c[0], c[1]) == (0, 0))
    assert p00 == pytest.approx(0.66 / 0.98, abs=1e-6)
    assert p00 > 0.5


def test_calibrate_score_probs_3plus_boosted():
    """历史 3+ 球占 50%，模型 top1 只有 31% → 校准后 3+ 合计应上升。"""
    top = [[1, 0, 0.40], [0, 0, 0.25], [2, 0, 0.20], [3, 1, 0.08], [2, 1, 0.07]]
    out = calibrate_score_probs(top, PRIOR, shrink=0.4)
    raw_3plus = sum(c[2] for c in top if c[0] + c[1] >= 3)
    cal_3plus = sum(c[2] for c in out if c[0] + c[1] >= 3)
    assert cal_3plus > raw_3plus


def test_calibrate_score_probs_missing_prior_returns_input():
    out = calibrate_score_probs([[1, 0, 0.5]], {"score": {}}, shrink=0.4)
    assert out == [[1, 0, 0.5]]


def test_calibrate_total_goals_shrinks():
    # 模型低估 3 球（10% vs 先验 22.1%）
    tg = [[0, 0.10], [1, 0.30], [2, 0.35], [3, 0.10], [4, 0.15]]
    out = calibrate_total_goals(tg, PRIOR, shrink=0.4)
    p3 = next(p for g, p in out if g == 3)
    assert p3 > 0.10  # 向 22.1% 先验收缩后上升
    assert abs(sum(p for _, p in out) - 1.0) < 1e-9


def test_empirical_score_prob():
    assert empirical_score_prob(1, 0, PRIOR) == 0.132
    assert empirical_score_prob(9, 9, PRIOR) == 0.001  # 地板
