"""北京时间工具 — 竞彩按北京时间开赛；GitHub Actions runner 是 UTC。

2026-08-14 事故：fetch 脚本直接 datetime.now()，在北京时间 00:00-08:00
会选错日期，导致 8/14 sina_odds 全缺。所有"今天/昨天"口径统一走这里，
避免三处各写各的 UTC+8 公式。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

BEIJING_TZ = timezone(timedelta(hours=8))


def beijing_now() -> datetime:
    """当前北京时间（带时区）。"""
    return datetime.now(BEIJING_TZ)


def beijing_today() -> str:
    """今天（北京时间）YYYY-MM-DD。"""
    return beijing_now().strftime("%Y-%m-%d")


def beijing_yesterday() -> str:
    """昨天（北京时间）YYYY-MM-DD。"""
    return (beijing_now() - timedelta(days=1)).strftime("%Y-%m-%d")
