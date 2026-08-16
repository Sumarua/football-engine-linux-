"""回归测试：串关/波胆复盘统计必须区分“已出票投入”和“已结算投入”。

2026-08-16 修复前：_stats() 只加总已结算票 stake，待结算票的 stake 被算成 0，
导致页面显示“出票5/0结算 · 投入¥0”，与逐票明细矛盾。
"""
from __future__ import annotations

import json

from engine.review.settle_parlays import build_settle_report


def _write_ticket_plan(day_dir, tickets):
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "ticket_plan.json").write_text(
        json.dumps({"parlay": tickets, "score_parlay": []}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_pending_tickets_stake_committed(tmp_path):
    daily = tmp_path / "data" / "daily"
    day = daily / "2026-08-16"
    _write_ticket_plan(day, [
        {
            "type": "2串1",
            "stake": 6.52,
            "potential": 28.6,
            "total_odds": 4.39,
            "note": "待赛",
            "legs": [
                {"match": f"{day.name}_周日001", "home": "A", "away": "B",
                 "sel": "home", "odds": 2.0},
                {"match": f"{day.name}_周日002", "home": "C", "away": "D",
                 "sel": "home", "odds": 2.2},
            ],
        },
        {
            "type": "2串1",
            "stake": 5.33,
            "potential": 22.46,
            "total_odds": 4.21,
            "note": "待赛",
            "legs": [
                {"match": f"{day.name}_周日003", "home": "E", "away": "F",
                 "sel": "away", "odds": 1.9},
                {"match": f"{day.name}_周日002", "home": "C", "away": "D",
                 "sel": "home", "odds": 2.2},
            ],
        },
    ])
    out = tmp_path / "parlay_settle.json"
    report = build_settle_report(daily_root=daily, out_path=out)

    day_stats = report["by_date"]["2026-08-16"]["parlay"]["stats"]
    assert day_stats["n_tickets"] == 2
    assert day_stats["n_pending"] == 2
    assert day_stats["n_settled"] == 0
    assert day_stats["stake"] == 0
    assert day_stats["stake_committed"] == 11.85
    assert day_stats["stake_pending"] == 11.85
    assert day_stats["roi"] is None

    # 全局统计同样不能把待结算投入丢掉
    assert report["parlay"]["stake_committed"] == 11.85
    assert report["parlay"]["stake"] == 0


def test_settled_and_pending_stake_split(tmp_path):
    daily = tmp_path / "data" / "daily"
    day = daily / "2026-08-15"
    # 已结算票需要赛果才能 settle；这里直接用已结算票（legs 带赛果命中）
    (day / "results.json").parent.mkdir(parents=True, exist_ok=True)
    (day / "results.json").write_text(json.dumps({
        "results": [
            {"match_id": f"{day.name}_周六001", "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 0},
        ]
    }), encoding="utf-8")
    _write_ticket_plan(day, [
        {
            "type": "2串1",
            "stake": 4.0,
            "potential": 10.0,
            "total_odds": 2.5,
            "note": "已结算",
            "legs": [
                {"match": f"{day.name}_周六001", "home": "A", "away": "B",
                 "sel": "home", "odds": 1.8},
                {"match": f"{day.name}_周六001", "home": "A", "away": "B",
                 "sel": "home", "odds": 1.4},
            ],
        },
        {
            "type": "3串1",
            "stake": 3.0,
            "potential": 9.0,
            "total_odds": 3.0,
            "note": "待赛",
            "legs": [
                {"match": f"{day.name}_周六002", "home": "X", "away": "Y",
                 "sel": "home", "odds": 1.5},
                {"match": f"{day.name}_周六003", "home": "P", "away": "Q",
                 "sel": "home", "odds": 2.0},
                {"match": f"{day.name}_周六004", "home": "R", "away": "S",
                 "sel": "home", "odds": 1.0},
            ],
        },
    ])
    out = tmp_path / "parlay_settle.json"
    report = build_settle_report(daily_root=daily, out_path=out)

    day_stats = report["by_date"]["2026-08-15"]["parlay"]["stats"]
    assert day_stats["n_tickets"] == 2
    assert day_stats["n_settled"] == 1
    assert day_stats["n_pending"] == 1
    assert day_stats["stake"] == 4.0
    assert day_stats["stake_committed"] == 7.0
    assert day_stats["stake_pending"] == 3.0
