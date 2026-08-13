"""盘口信号方向回归测试（engine.market_signal）。

锁定 2026-08-14 事故：落盘 market_signal 曾用反向公式 (1-c)*2 —— c>1（资金
涌入）给负分，与"应用侧 >1.05 加仓"相反，导致结算统计的"盘口信号命中率"
方向是反的。本测试确保口径永不再被改回。
"""
from __future__ import annotations

from engine.market_signal import (
    compression_signals,
    compression_signal,
    signal_strength_class,
)


def test_compression_signal_sign_convention():
    """compression = 初盘/即时盘；>1 = 资金涌入 → 正信号。"""
    assert compression_signal(1.05) == 0.10
    assert compression_signal(1.10) == 0.20
    assert compression_signal(0.95) == -0.10
    assert compression_signal(0.90) == -0.20
    assert compression_signal(1.0) == 0.0


def test_compression_signal_none_and_garbage():
    assert compression_signal(None) == 0.0
    assert compression_signal(1.1234) == 0.2468  # round 4 位


def test_compression_signals_maps_three_directions():
    out = compression_signals({"home": 1.10, "away": 0.90})
    assert out == {"home": 0.20, "draw": 0.0, "away": -0.20}


def test_compression_signals_missing_dict():
    assert compression_signals(None) == {"home": 0.0, "draw": 0.0, "away": 0.0}


def test_signal_strength_class():
    assert signal_strength_class(0.10) == "加仓"
    assert signal_strength_class(-0.15) == "减仓"
    assert signal_strength_class(0.0) == "持平"
