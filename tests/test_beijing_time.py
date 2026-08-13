"""北京时间工具回归测试（engine.beijing_time）。

锁定 2026-08-14 事故：fetch 脚本直接 datetime.now()（runner 是 UTC），
在北京时间 00:00-08:00 会选错日期 → 8/14 sina_odds 全缺。统一口径后，
用 mock 验证"UTC 凌晨 = 北京当天"的边界。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest import mock

import engine.beijing_time as bt


def test_beijing_offset_is_plus_8():
    assert bt.beijing_now().utcoffset() == timedelta(hours=8)


def test_beijing_today_format():
    d = bt.beijing_today()
    assert len(d) == 10
    datetime.strptime(d, "%Y-%m-%d")  # 解析成功即格式正确


def test_beijing_yesterday_is_today_minus_one():
    assert bt.beijing_yesterday() == (
        datetime.strptime(bt.beijing_today(), "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")


def test_utc_evening_is_beijing_next_day():
    """UTC 2026-08-13 18:30 = 北京 2026-08-14 02:30（关键边界：日期不同）。"""
    fixed_utc = datetime(2026, 8, 13, 18, 30, tzinfo=timezone.utc)
    with mock.patch.object(bt, "beijing_now", return_value=fixed_utc.astimezone(bt.BEIJING_TZ)):
        assert bt.beijing_today() == "2026-08-14"
        assert bt.beijing_yesterday() == "2026-08-13"


def test_utc_noon_is_same_beijing_day():
    """UTC 2026-08-14 04:00 = 北京 2026-08-14 12:00（日期相同）。"""
    fixed_utc = datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc)
    with mock.patch.object(bt, "beijing_now", return_value=fixed_utc.astimezone(bt.BEIJING_TZ)):
        assert bt.beijing_today() == "2026-08-14"
