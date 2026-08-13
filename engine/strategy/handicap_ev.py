"""让球胜平负 EV 评估 — 竞彩让球玩法的价值挖掘

原理：
- 模型已算出比分联合分布（top_scores: [主队进球, 客队进球, 概率]）
- 竞彩让球盘（hhad）：主队让出/受让 handicap 球后，胜平负重新划分
- 让球后概率 = 比分分布按 (主-客+handicap) 符号加总
- 对比官方让球赔率（handicap_home_odds 等）→ 找正 EV 场次

竞彩让球赔率通常接近 2.0（均衡盘），比胜平负大热盘（1.2-1.4）
更容易覆盖水钱，是模型优势最值得变现的玩法。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# 比分/总进球经验先验（懒加载，缓存）：校准模型比分矩阵的低比分系统性偏差。
# 与 multi_play_ev 同款实现，避免每次评估都重读 JSON。
_prior_cache: dict | None = None


def _score_prior() -> dict:
    global _prior_cache
    if _prior_cache is None:
        try:
            from engine.learning.score_calibration import load_score_prior
            _prior_cache = load_score_prior()
        except Exception:
            _prior_cache = {}
    return _prior_cache


@dataclass
class HandicapEV:
    """一场比赛的让球 EV 评估"""
    match_id: str
    home_team: str
    away_team: str
    handicap: float                # 让球线（主队视角，正=主让）
    probs: dict                    # 让球后胜平负概率 {home, draw, away}
    odds: dict                     # 官方让球赔率 {home, draw, away}
    edges: dict = field(default_factory=dict)   # 各方向 edge
    market_edge: float = 0.0                     # 市场去水口径的 best_sel edge
    best_sel: str = ""
    best_edge: float = 0.0
    ev: float = 0.0
    recommended: bool = False


def handicap_probs_from_scores(
    top_scores: list, handicap: float
) -> dict | None:
    """从比分分布推导让球后胜平负概率。

    top_scores: [[主队进球, 客队进球, 概率], ...]
    handicap: 让球线（竞彩口径：主队让 handicap 球，胜 = 主队进球 - 客队进球 + handicap > 0）
    """
    if not top_scores or handicap is None:
        return None
    ph = pd_ = pa = 0.0
    for row in top_scores:
        try:
            hs, as_, p = row[0], row[1], row[2]
        except (IndexError, TypeError, ValueError):
            continue
        diff = hs - as_ + handicap
        if diff > 0:
            ph += p
        elif diff == 0:
            pd_ += p
        else:
            pa += p
    total = ph + pd_ + pa
    if total <= 0:
        return None
    return {"home": ph / total, "draw": pd_ / total, "away": pa / total}


def evaluate_handicap_ev(
    pred: dict, min_edge: float = 0.03, market_weight: float = 0.882
) -> HandicapEV | None:
    """评估一场预测的让球 EV（独立预测：让球方向可与胜平负不同）。

    概率口径与 1X2 融合一致——市场主导。1X2 融合冠军权重 model=0.10 /
    market=0.75 / djyy=0.15，让球无 djyy 源，故归一化为 model:market =
    0.10:0.75 ≈ 0.118:0.882（market_weight=0.882）。纯模型比分矩阵（Brier
    弱于随机）曾"对抗市场"押受让方，让球回测 83 场 ROI -18.1%。
    fused = (1-w)*模型让球概率 + w*市场去水让球概率，再用 fused 算 edge。
    """
    handicap = pred.get("handicap")
    odds = {
        "home": pred.get("handicap_home_odds"),
        "draw": pred.get("handicap_draw_odds"),
        "away": pred.get("handicap_away_odds"),
    }
    if handicap is None or not any(o is not None and o > 1.0 for o in odds.values()):
        return None

    # 模型让球概率（2026-08-14：优先从"校准后"的比分矩阵推导——比分矩阵是
    # 让球概率的源，先做贝叶斯收缩（score_calibration, shrink=0.4，修正低比分
    # 系统性偏差）才算"修复"而非"压制"；handicap_home_prob 等是预测时从原始
    # 矩阵算的，未校准，仅作 top_scores 缺失时的回退）。
    raw_scores = pred.get("top_scores")
    model_probs = None
    if raw_scores:
        try:
            from engine.learning.score_calibration import calibrate_score_probs
            raw_scores = calibrate_score_probs(raw_scores, _score_prior())
        except Exception:
            pass
        model_probs = handicap_probs_from_scores(raw_scores, handicap)
    if not model_probs:
        mprobs = {
            "home": pred.get("handicap_home_prob"),
            "draw": pred.get("handicap_draw_prob"),
            "away": pred.get("handicap_away_prob"),
        }
        if mprobs and all(p is not None for p in mprobs.values()) and 0.9 < sum(mprobs.values()) <= 1.05:
            model_probs = mprobs
    if not model_probs:
        return None

    # 市场去水让球概率（Shin 在竞彩 1.128 高水位下退化为乘法归一化，已实证）
    _implied = [1.0 / (odds[s] or 1.0) for s in ("home", "draw", "away")]
    _implied_total = sum(_implied)
    market_probs = {
        "home": _implied[0] / _implied_total,
        "draw": _implied[1] / _implied_total,
        "away": _implied[2] / _implied_total,
    } if _implied_total > 0 else None

    # 市场主导融合（与 1X2 同口径）；无市场概率时退回纯模型
    if market_probs and 0 < market_weight <= 1.0:
        probs = {
            s: (1 - market_weight) * model_probs[s] + market_weight * market_probs[s]
            for s in ("home", "draw", "away")
        }
        _total = sum(probs.values())
        if _total > 0:
            probs = {s: v / _total for s, v in probs.items()}
    else:
        probs = model_probs

    ev = HandicapEV(
        match_id=pred.get("match_id", ""),
        home_team=pred.get("home_team", ""),
        away_team=pred.get("away_team", ""),
        handicap=handicap,
        probs=probs,
        odds=odds,
    )
    best_sel, best_edge = "", -1.0
    for sel in ("home", "draw", "away"):
        o = odds[sel]
        if o is None or o <= 1.0:
            continue  # 只评估有赔率的方向
        edge = probs[sel] * o - 1.0
        ev.edges[sel] = edge
        if edge > best_edge:
            best_edge, best_sel = edge, sel
    ev.best_sel, ev.best_edge, ev.ev = best_sel, best_edge, best_edge

    # 市场口径 edge（best_sel 方向，用于展示/复盘）
    if market_probs and best_sel in odds and odds[best_sel]:
        ev.market_edge = market_probs[best_sel] * odds[best_sel] - 1.0
    else:
        ev.market_edge = -1.0

    # sanity check + 方向一致性闸：fused edge 严重背离(>30%) 多为脏数据；
    # 且模型让球方向必须与市场让球方向一致才推荐（不让球/让球方向可以不同，
    # 但"下注"不能赌"模型比市场更懂"——让球回测 83 场 ROI -18.1% 全在对抗市场）。
    _model_dir = max(model_probs, key=model_probs.get) if model_probs else ""
    _market_dir = max(market_probs, key=market_probs.get) if market_probs else ""
    _dir_agree = (_model_dir == _market_dir) if (_model_dir and _market_dir) else True
    ev.recommended = (best_edge >= min_edge and best_edge <= 0.30) and _dir_agree
    return ev


def scan_handicap_ev(
    predictions: list[dict], min_edge: float = 0.03
) -> list[HandicapEV]:
    """扫描全部预测，返回有让球数据的场次 EV 评估。"""
    out = []
    for p in predictions:
        ev = evaluate_handicap_ev(p, min_edge=min_edge)
        if ev:
            out.append(ev)
    return out


def build_handicap_report(
    daily_root: Path | None = None, out_path: Path | None = None
) -> dict:
    """扫描所有 daily 目录已结算场次，回测让球玩法 ROI。"""
    daily_root = daily_root or Path(__file__).parent.parent.parent / "data" / "daily"
    out_path = out_path or daily_root.parent / "state" / "handicap_report.json"

    rows = []
    for f in sorted(daily_root.glob("*/predictions.json")):
        try:
            preds = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for p in preds:
            ah, aa = p.get("actual_home_score"), p.get("actual_away_score")
            if ah is None or aa is None:
                continue
            ev = evaluate_handicap_ev(p, min_edge=-1.0)
            if not ev:
                continue
            actual = "home" if ah - aa + ev.handicap > 0 else (
                "draw" if ah - aa + ev.handicap == 0 else "away")
            hit = ev.best_sel == actual
            pnl = (ev.odds[ev.best_sel] - 1) if hit else -1.0
            rows.append({
                "date": f.parent.name,
                "match_id": p.get("match_id", ""),
                "league": p.get("competition", ""),
                "handicap": ev.handicap,
                "best_sel": ev.best_sel,
                "best_edge": round(ev.best_edge, 4),
                "actual": actual,
                "hit": hit,
                "odds": ev.odds[ev.best_sel],
                "pnl": round(pnl, 4),
            })

    report = {
        "n_matches": len(rows),
        "hits": sum(1 for r in rows if r["hit"]),
        "hit_rate": round(sum(1 for r in rows if r["hit"]) / len(rows), 4) if rows else 0,
        "roi": round(sum(r["pnl"] for r in rows) / len(rows), 4) if rows else 0,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    import sys
    from engine.main import load_config
    r = build_handicap_report()
    print(f"让球玩法回测: {r['n_matches']} 场, 命中率 {r['hit_rate']*100:.1f}%, ROI {r['roi']*100:+.1f}%")
