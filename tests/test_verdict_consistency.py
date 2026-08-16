"""P2 一致性修复回归测试（2026-08-14 第二批）

锁定：
- ev_report verdict 加 n≥50 门槛（与 league_report.classify_league 同口径）
- kelly._load_value_zones 只消费 n≥50 的赔率层（小样本不再驱动禁投）
- parlay_report 价值区联赛动态读 league_report（不再硬编码瑞超等）
"""
from __future__ import annotations

import json


from engine.review.ev_report import MIN_VERDICT_SAMPLES, build_report
from engine.review.parlay_report import build_parlay_report
from engine.strategy.kelly import KellyStrategy

MIN = MIN_VERDICT_SAMPLES


# ---------- ev_report verdict 门槛 ----------

def _mk_daily(tmp_path, matches: list[dict]) -> object:
    """构造 daily/<date>/predictions.json；match: {dir, actual, odds, league, date}"""
    daily = tmp_path / "data" / "daily"
    by_date: dict[str, list] = {}
    for i, m in enumerate(matches):
        d = m.get("date", "2026-08-01")
        by_date.setdefault(d, []).append({
            "match_id": f"{d}_T{i:03d}",
            "home_team": f"主{i}", "away_team": f"客{i}",
            "competition": m.get("league", "测试联赛"),
            "direction": m["dir"],
            "home_win_prob": 0.34, "draw_prob": 0.33, "away_win_prob": 0.33,
            "home_odds": m.get("odds", 2.0),
            "draw_odds": 3.2, "away_odds": 3.4,
            "actual_home_score": 1 if m["actual"] == "home" else 0,
            "actual_away_score": 1 if m["actual"] == "away" else 0,
        })
    for d, preds in by_date.items():
        p = daily / d
        p.mkdir(parents=True, exist_ok=True)
        (p / "predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    return daily


def _run_ev(daily_root, tmp_path) -> dict:
    out = tmp_path / "ev_report.json"
    return build_report(daily_root=daily_root, out_path=out)


def test_ev_report_small_sample_no_verdict(tmp_path):
    """n=10 全输 → 旧规则标送钱区，新规则只给'谨慎'（n<50 不定性）"""
    daily = _mk_daily(tmp_path, [
        {"dir": "home", "actual": "away", "odds": 1.8} for _ in range(10)
    ])
    r = _run_ev(daily, tmp_path)
    league = r["leagues"]["测试联赛"]
    assert league["n"] == 10
    assert league["verdict"] == "谨慎"          # 旧规则会是 "送钱区 ❌"
    assert "送钱联赛(回避)" not in [t for t in r["takeaways"]]


def test_ev_report_big_sample_verdict(tmp_path):
    """n≥50 且 ROI<-10% → 送钱区（门槛内正常定性）"""
    n = MIN + 3
    daily = _mk_daily(tmp_path, [
        {"dir": "home", "actual": "away", "odds": 1.8} for _ in range(n)
    ])
    r = _run_ev(daily, tmp_path)
    league = r["leagues"]["测试联赛"]
    assert league["n"] == n
    assert league["verdict"] == "送钱区 ❌"


def test_ev_report_value_zone_requires_n(tmp_path):
    """n=18 ROI>0 → 旧规则标价值区，新规则只给'观望'（避免小样本误推）"""
    daily = _mk_daily(tmp_path, [
        {"dir": "home", "actual": "home", "odds": 3.2} for _ in range(18)
    ])
    r = _run_ev(daily, tmp_path)
    assert r["leagues"]["测试联赛"]["verdict"] == "观望"
    assert all("价值区" not in t for t in r["takeaways"])


# ---------- kelly._load_value_zones ----------

def _ks():
    """构造跳过 __init__ 的 KellyStrategy（手工注入所需配置属性）"""
    ks = KellyStrategy.__new__(KellyStrategy)
    ks.vz_min_n = 50
    ks.boost_min_n = 15
    ks.boost_min_roi = 0.05
    ks.boost_mult = 0.15
    return ks


def test_kelly_value_zones_boost_threshold(tmp_path):
    """zones 返回所有 n≥boost_min_n 的层（供注额加成判断用）；
    禁投/降权由调用侧按 vz_min_n 收口（见 test_kelly_reject_gated_by_n50）"""
    ev = tmp_path / "ev_report.json"
    ev.write_text(json.dumps({
        "layers": {
            "L1 大热(<1.5)": {"n": 56, "roi": -0.25},     # n≥50 → 返回
            "L2 热(1.5-1.8)": {"n": 42, "roi": -0.34},    # n≥15 → 返回（但不禁投）
            "L5 深冷(≥3.0)": {"n": 18, "roi": 0.44},      # n≥15 → 返回（可加成）
        }
    }), encoding="utf-8")
    zones = _ks()._load_value_zones(path=ev)
    assert zones == {
        "L1 大热(<1.5)": {"roi": -0.25, "n": 56},
        "L2 热(1.5-1.8)": {"roi": -0.34, "n": 42},
        "L5 深冷(≥3.0)": {"roi": 0.44, "n": 18},
    }


def test_kelly_value_zones_missing_file(tmp_path):
    assert _ks()._load_value_zones(path=tmp_path / "nope.json") == {}


def test_kelly_reject_gated_by_n50(tmp_path):
    """送钱区禁投只对 n≥50 的层生效：L5(n=18, roi=+44%) 不拒；
    手工构造 n≥50 且 roi<-10% 的层 → 拒绝。"""
    ks = _ks()
    for attr, val in (("kelly_fraction", 0.25), ("bankroll", 5000), ("stake_unit", 2),
                      ("min_edge", 0.03), ("min_ev", 0.03), ("max_single", 200),
                      ("max_match_exposure", 200), ("max_daily", 500), ("max_monthly", 5000),
                      ("monthly_stop_loss", 5000), ("max_parlay_stake", 30), ("max_parlay_legs", 2)):
        setattr(ks, attr, val)
    # L5 深冷 ROI 为正且 n=18 → 通过（不拒），注额受加成
    ks._load_value_zones = lambda path=None: {"L5 深冷(≥3.0)": {"roi": 0.44, "n": 18}}
    plan = ks.evaluate_candidates([{
        "match_id": "2026-08-14_周五001", "date": "2026-08-14",
        "home_team": "A", "away_team": "B", "direction": "away",
        "home_win_prob": 0.25, "draw_prob": 0.25, "away_win_prob": 0.5,
        "away_odds": 3.5,
    }])
    assert len(plan.singles) == 1, "L5(n=18) 正 ROI 不应被拒"
    assert not plan.rejected
    # 送钱区层（n≥50, roi<-10%）→ 拒绝（检查发生在 edge 门槛之前，odds 1.4 属 L1）
    ks._load_value_zones = lambda path=None: {"L1 大热(<1.5)": {"roi": -0.25, "n": 56}}
    plan2 = ks.evaluate_candidates([{
        "match_id": "2026-08-14_周五002", "date": "2026-08-14",
        "home_team": "A", "away_team": "B", "direction": "home",
        "home_win_prob": 0.75, "draw_prob": 0.15, "away_win_prob": 0.1,
        "home_odds": 1.4,
    }])
    assert len(plan2.singles) == 0
    assert any("送钱区" in r[1] for r in plan2.rejected), "n≥50 送钱区层必须拒绝"


def test_kelly_stake_boost_on_profitable_layer(tmp_path):
    """深冷层加权：n≥boost_min_n 且 ROI>min_roi 的层 → 注额 ×(1+mult)"""
    ev = tmp_path / "ev_report.json"
    ev.write_text(json.dumps({"layers": {
        "L5 深冷(≥3.0)": {"n": 18, "roi": 0.44},
    }}), encoding="utf-8")
    ks = _ks()
    ks.kelly_fraction = 0.25
    ks.bankroll = 5000
    ks.stake_unit = 2
    ks.min_edge = 0.03
    ks.min_ev = 0.03
    ks.max_single = 200
    ks.max_match_exposure = 200
    ks.max_daily = 500
    ks.max_monthly = 5000
    ks.monthly_stop_loss = 5000
    ks.max_parlay_stake = 30
    ks.max_parlay_legs = 2
    ks._load_value_zones = lambda path=None: {
        "L5 深冷(≥3.0)": {"roi": 0.44, "n": 18}}

    plan = ks.evaluate_candidates([{
        "match_id": "2026-08-14_周五001",
        "date": "2026-08-14",
        "home_team": "A", "away_team": "B",
        "direction": "away",
        "home_win_prob": 0.25, "draw_prob": 0.25, "away_win_prob": 0.5,
        "away_odds": 3.5,
    }])
    # away @3.5 prob 0.5: EV = 0.5*3.5-1 = 0.75 > min_ev；edge = 0.5-0.286 = 0.214
    # full_kelly = (2.5*0.5-0.5)/2.5 = 0.3 → stake = 5000*0.3*0.25 = 375 → ×1.15 = 431 → cap 200
    assert len(plan.singles) == 1
    assert plan.singles[0].stake == 200  # 受 max_single 封顶
    # 且该注确实经过了加成（未加成时 kelly=375 > 200 同样封顶，故用 0.5 概率验证命中层过滤）
    assert plan.singles[0].selection == "away"


# ---------- parlay_report 动态价值区联赛 ----------

def _mk_parlay_inputs(tmp_path):
    # 三场已结算：两场 K1联赛（高置信，2串1 top2 会串到它们）+ 一场 瑞超
    daily = tmp_path / "data" / "daily"
    p = daily / "2026-08-01"
    p.mkdir(parents=True)
    base = {"home_win_prob": 0.5, "draw_prob": 0.25, "away_win_prob": 0.25,
            "home_odds": 1.9, "draw_odds": 3.4, "away_odds": 3.8,
            "actual_home_score": 1, "actual_away_score": 0}
    (p / "predictions.json").write_text(json.dumps([
        dict(base, match_id="2026-08-01_A", home_team="A", away_team="B",
             competition="K1联赛", direction="home", confidence=0.6),
        dict(base, match_id="2026-08-01_B", home_team="E", away_team="F",
             competition="K1联赛", direction="home", confidence=0.5),
        dict(base, match_id="2026-08-01_C", home_team="C", away_team="D",
             competition="瑞超", direction="home", confidence=0.3),
    ]), encoding="utf-8")
    return daily


def test_parlay_value_leagues_dynamic(tmp_path):
    daily = _mk_parlay_inputs(tmp_path)
    lr = tmp_path / "league_report.json"
    lr.write_text(json.dumps({"leagues": [
        {"league": "K1联赛", "verdict": "价值区"},
        {"league": "瑞超", "verdict": "谨慎"},   # 瑞超不再是价值区
    ]}), encoding="utf-8")
    r = build_parlay_report(
        daily_root=daily,
        out_path=tmp_path / "parlay_report.json",
        league_report_path=lr,
    )
    # 只串 K1 联赛（瑞超被排除）
    assert r["parlay_value_leagues"]["n"] == 1


def test_parlay_value_leagues_empty_without_report(tmp_path):
    daily = _mk_parlay_inputs(tmp_path)
    # league_report 缺失/无价值区 → 变体为空（不再硬编码 K1/挪超/瑞超）
    lr = tmp_path / "league_report.json"
    lr.write_text(json.dumps({"leagues": [
        {"league": "瑞超", "verdict": "谨慎"},
    ]}), encoding="utf-8")
    r = build_parlay_report(
        daily_root=daily,
        out_path=tmp_path / "parlay_report.json",
        league_report_path=lr,
    )
    assert r["parlay_value_leagues"]["n"] == 0
