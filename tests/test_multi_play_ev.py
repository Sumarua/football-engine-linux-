"""多玩法赔率解析回归测试。

重点锁定 2026-08-14 事故：sporttery 返回的 'f' 后缀键（s0f/s00s00f，值 0/1
的解析标志）曾被当成真实赔率，0 覆盖了真实赔率 → 波胆/总进球 EV 静默全灭。
"""
from __future__ import annotations

from engine.strategy.multi_play_ev import (
    merge_over7,
    norm_scores,
    norm_total_goals,
    parse_crs_odds,
    parse_hafu_odds,
    parse_ttg_odds,
)


# ---------- f 键回归 ----------

def test_parse_ttg_skips_f_flag_keys():
    """s0f 是解析标志不是赔率：修复前 0 会覆盖 s0=8.00，总进球 EV 全灭。"""
    raw = {"s0": 8.0, "s0f": 0, "s1": 5.5, "s1f": 1, "s2": 4.2, "s2f": 0}
    out = parse_ttg_odds(raw)
    assert out == {0: 8.0, 1: 5.5, 2: 4.2}


def test_parse_ttg_s7_means_7plus():
    out = parse_ttg_odds({"s7": 12.0, "s7f": 0})
    assert out == {7: 12.0}


def test_parse_crs_skips_f_flag_keys():
    """s00s00f 等标志键跳过，否则 0/1 会覆盖真实波胆赔率。"""
    raw = {
        "s00s00": 15.0, "s00s00f": 1,
        "s01s00": 12.0, "s01s00f": 0,
        "s00s01": 9.5, "s00s01f": 0,
    }
    out = parse_crs_odds(raw)
    assert out == {(0, 0): 15.0, (1, 0): 12.0, (0, 1): 9.5}


def test_parse_crs_colon_format():
    out = parse_crs_odds({"1:0": 9.0, "2:1": 11.0})
    assert out == {(1, 0): 9.0, (2, 1): 11.0}


def test_parse_crs_plain_digits():
    out = parse_crs_odds({"0100": 21.0})  # 前两位主后两位客
    assert out == {(1, 0): 21.0}


def test_parse_crs_empty_and_garbage():
    assert parse_crs_odds(None) == {}
    assert parse_crs_odds({"notanumber": 5.0}) == {}


def test_parse_hafu_normalizes_upper():
    out = parse_hafu_odds({"hh": 3.2, "HD": 5.0, "da": 8.0, "xx": 9.9})
    assert out == {"HH": 3.2, "HD": 5.0, "DA": 8.0}


# ---------- 归一化 ----------

def test_norm_scores():
    out = norm_scores([[1, 0, 0.6], [0, 0, 0.2], [2, 1, 0.2]])
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out[(1, 0)] == 0.6


def test_norm_total_goals_merges_duplicates():
    out = norm_total_goals([[0, 0.1], [1, 0.3], [1, 0.2]])
    # 去重合并后归一化: 1球 = (0.3+0.2)/(0.1+0.5) = 0.5/0.6
    assert abs(out[1] - (0.5 / 0.6)) < 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_merge_over7():
    out = merge_over7({0: 0.1, 6: 0.2, 7: 0.3, 8: 0.4})
    assert out[7] == 0.7  # 7 与 7+ 合并
    assert 8 not in out
