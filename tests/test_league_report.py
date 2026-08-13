"""联赛 verdict 判定回归测试（engine.review.league_report.classify_league）。

锁定 2026-08-14 规则：送钱区/价值区定性需要累计 n≥50（此前 n=10-24 时
噪声主导，翻来翻去误禁好联赛）；小样本只给谨慎/观望。
"""
from __future__ import annotations

from engine.review.league_report import classify_league


def test_n_under_3_is_insufficient():
    assert classify_league(2, -0.3, 0.0, 2, 0.0, 2, 0.0) == "样本不足"


def test_small_sample_never_flips_hard_verdict():
    """n=10 且 roi 很差/很好 → 都不给送钱区/价值区，只给谨慎/观望。"""
    assert classify_league(10, -0.3, 0.2, 5, 0.2, 10, 0.2) == "谨慎"
    assert classify_league(10, 0.3, 0.8, 5, 0.8, 10, 0.8) == "观望"


def test_small_sample_positive_roi_is_watch():
    assert classify_league(24, 0.1, 0.6, 5, 0.8, 10, 0.6) == "观望"


def test_n50_bad_roi_is_money_pit():
    assert classify_league(50, -0.12, 0.35, 5, 0.2, 10, 0.3) == "送钱区"


def test_n50_good_roi_is_value():
    assert classify_league(60, 0.08, 0.55, 5, 0.6, 10, 0.5) == "价值区"


def test_n50_recovery_unban_requires_both_windows():
    """累计口径送钱区，但近5≥60% 且 近10≥50% → 回暖解禁。"""
    assert classify_league(60, -0.12, 0.35, 5, 0.8, 10, 0.6) == "回暖解禁"
    # 只满足近5（3中2=67%）不满足近10（4中6=40%）→ 不解禁（2026-08-13 双窗口）
    assert classify_league(60, -0.12, 0.35, 3, 0.667, 10, 0.4) == "送钱区"


def test_n50_neutral_is_cautious_or_watch():
    assert classify_league(50, -0.02, 0.45, 5, 0.4, 10, 0.45) == "谨慎"
    assert classify_league(50, 0.02, 0.45, 5, 0.4, 10, 0.45) == "观望"
