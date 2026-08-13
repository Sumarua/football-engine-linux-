"""让球 EV 回归测试（engine.strategy.handicap_ev）。

锁定：
1. handicap_probs_from_scores 的让球后胜平负重划分数学
2. evaluate_handicap_ev 从"校准后比分矩阵"推导模型概率（2026-08-14 修复）
3. 方向一致性闸：模型与市场让球方向背离时不推荐
"""
from __future__ import annotations

import pytest

import engine.strategy.handicap_ev as hev
from engine.strategy.handicap_ev import (
    evaluate_handicap_ev,
    handicap_probs_from_scores,
)


@pytest.fixture(autouse=True)
def _reset_prior_cache():
    """_score_prior() 有模块级缓存，测试间重置，避免 monkeypatch 失效。"""
    hev._prior_cache = None
    yield
    hev._prior_cache = None


def test_handicap_probs_from_scores_handicap1():
    top = [[2, 0, 0.5], [1, 1, 0.3], [0, 2, 0.2]]
    out = handicap_probs_from_scores(top, handicap=1)
    # 让1球后: 2-0+1=3>0(主胜), 1-1+1=1>0(主胜), 0-2+1=-1<0(客胜)
    assert out == {"home": 0.8, "draw": 0.0, "away": 0.2}


def test_handicap_probs_from_scores_handicap_minus1():
    top = [[2, 0, 0.5], [1, 1, 0.3], [0, 2, 0.2]]
    out = handicap_probs_from_scores(top, handicap=-1)
    # 受让1球后: 2-0-1=1>0(主胜), 1-1-1=-1<0(客胜), 0-2-1=-3<0(客胜)
    assert out == {"home": 0.5, "draw": 0.0, "away": 0.5}


def test_handicap_probs_empty_returns_none():
    assert handicap_probs_from_scores([], 1.0) is None
    assert handicap_probs_from_scores(None, 1.0) is None


def test_evaluate_handicap_ev_uses_calibrated_score_matrix(monkeypatch):
    """2026-08-14 修复：模型概率应从校准后的比分矩阵推导（先验收缩）。"""
    import engine.learning.score_calibration as sc
    monkeypatch.setattr(
        sc, "load_score_prior",
        lambda: {"score": {"1-0": 0.132, "2-0": 0.068, "0-2": 0.054}},
    )
    pred = {
        "match_id": "m1", "home_team": "A", "away_team": "B",
        "handicap": 1.0,
        "handicap_home_odds": 2.10, "handicap_draw_odds": 3.40, "handicap_away_odds": 3.10,
        # 只有比分矩阵，无 handicap_home_prob → 走校准推导路径
        "top_scores": [[2, 0, 0.50], [1, 1, 0.30], [0, 2, 0.20]],
    }
    ev = evaluate_handicap_ev(pred, min_edge=-1.0)
    assert ev is not None
    assert abs(sum(ev.probs.values()) - 1.0) < 1e-6
    # 让1球后主胜概率最大（模型+市场同向），best_sel=home
    assert ev.best_sel == "home"


def test_direction_gate_blocks_model_against_market(monkeypatch):
    """模型让球方向与市场相反 → 即使模型 edge 高也不推荐（不做对抗市场的单）。"""
    import engine.learning.score_calibration as sc
    monkeypatch.setattr(sc, "load_score_prior", lambda: {"score": {"0-2": 0.30}})
    pred = {
        "match_id": "m2", "home_team": "A", "away_team": "B",
        "handicap": 0.0,  # 平手盘
        # 市场强烈看主胜（低赔），模型强烈看客胜（比分矩阵）→ 方向背离
        "handicap_home_odds": 1.40, "handicap_draw_odds": 3.40, "handicap_away_odds": 3.50,
        "top_scores": [[0, 2, 0.98], [1, 0, 0.02]],
    }
    ev = evaluate_handicap_ev(pred, min_edge=0.03)
    assert ev is not None
    # 模型+市场融合后客胜 edge 仍为正（>3%），但方向背离 → 必须被闸住
    assert ev.best_sel == "away"
    assert ev.recommended is False
