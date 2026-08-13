"""盘口信号（水位）纯函数 — 赔率压缩比 → 方向信号分

口径（与 scripts/fetch_sina_odds.py 一致）：
    compression = 初盘赔率 / 即时赔率
    > 1    = 赔率被压缩（资金涌入，市场看好该方向）→ 正信号
    < 1    = 赔率被抬高（资金撤出，市场看衰该方向）→ 负信号

2026-08-14 复盘：应用侧（main.py 概率修正，>1.05 加仓 / <0.95 减仓）已按
此口径修复，但**落盘 market_signal 仍用旧的反向公式 (1-c)*2**——结算统计的
"盘口信号命中率"方向因此是反的。本模块统一口径，并供回归测试锁定，防止
再被改回反向约定。
"""
from __future__ import annotations


def compression_signal(compression: float | None) -> float:
    """压缩比 → 方向信号分（四舍五入到 4 位，与预测落盘精度一致）。

    c=1.05（资金涌入）→ +0.10；c=0.95（资金撤出）→ -0.10；c=1.0 → 0。
    """
    if compression is None:
        return 0.0
    return round((float(compression) - 1.0) * 2.0, 4)


def compression_signals(compression: dict | None) -> dict:
    """三方向压缩比 → 三方向信号分。缺失方向按 1.0（无变化）处理。"""
    out = {}
    for k in ("home", "draw", "away"):
        _c = (compression or {}).get(k, 1.0)
        out[k] = compression_signal(_c if _c is not None else 1.0)
    return out


def signal_strength_class(signal: float) -> str:
    """信号分 → 强度档位（页面展示用）。"""
    if signal >= 0.10:
        return "加仓"
    if signal <= -0.10:
        return "减仓"
    return "持平"
