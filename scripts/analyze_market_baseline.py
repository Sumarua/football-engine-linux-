#!/usr/bin/env python3
"""纯市场基线分析（2026-08-14，P1 诊断）

问题：系统整体 ROI -18.9%（201 场命中 43.3%），Brier 融合 0.655 > 市场 0.629。
本脚本回答：模型/融合到底有没有给市场赔率加分？

方法：对全部已结算预测，同一批样本上对比三个"概率源"——
  market:  market_fair（市场公平概率）
  model:   model_raw（模型原始概率）
  fused:   home_win_prob/draw_prob/away_win_prob（生产融合概率，系统实际所用）
指标（每个源各自独立计算，互不干扰）：
  hit_rate:  argmax 方向命中率
  brier:     3 分类 Brier（越低越好；市场 0.25 = 随机三选一）
  roi_always: 每场 1 单位押 argmax 方向（该方向赔率，(odds-1) 赢 / -1 输）
  roi_edge:   仅当 prob > 1/odds（正期望）才押的 ROI（近似 Kelly 门槛行为）
  by_layer:   按赔率层 L1-L5 拆的 roi_always（验证"全层送钱"是否各源一致）

产出：data/state/market_baseline.json + 控制台对比表
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent


def layer_of(odds: float) -> str:
    if odds < 1.5:
        return "L1 大热(<1.5)"
    if odds < 1.8:
        return "L2 热(1.5-1.8)"
    if odds < 2.2:
        return "L3 中(1.8-2.2)"
    if odds < 3.0:
        return "L4 冷(2.2-3.0)"
    return "L5 深冷(≥3.0)"


def _source_probs(p: dict, key: str):
    """返回 (home, draw, away) 或 None"""
    if key == "market":
        mf = p.get("market_fair")
        if isinstance(mf, list) and len(mf) == 3:
            return tuple(float(x) for x in mf)
        return None
    if key == "model":
        mr = p.get("model_raw")
        if isinstance(mr, dict) and all(k in mr for k in ("home", "draw", "away")):
            return (float(mr["home"]), float(mr["draw"]), float(mr["away"]))
        return None
    # fused
    probs = (p.get("home_win_prob"), p.get("draw_prob"), p.get("away_win_prob"))
    if all(x is not None for x in probs):
        return tuple(float(x) for x in probs)
    return None


def _actual(p: dict):
    ah, aa = p.get("actual_home_score"), p.get("actual_away_score")
    if ah is None or aa is None:
        return None
    return "home" if ah > aa else ("draw" if ah == aa else "away")


def _odds(p: dict, sel: str):
    return p.get(f"{sel}_odds")


def load_settled() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "daily" / "*" / "predictions.json"))):
        try:
            preds = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for p in preds:
            actual = _actual(p)
            if actual is None:
                continue
            rows.append({
                "date": Path(f).parent.name,
                "league": p.get("competition", "?"),
                "actual": actual,
                "system_direction": p.get("direction"),
                "conf": p.get("confidence", 0) or 0,
                "odds": {s: _odds(p, s) for s in ("home", "draw", "away")},
                "market": _source_probs(p, "market"),
                "model": _source_probs(p, "model"),
                "fused": _source_probs(p, "fused"),
            })
    return rows


def _stats_for(rows: list[dict], source_key: str) -> dict:
    n = hit = brier_sum = 0.0
    roi_always = roi_edge = 0.0
    n_edge = 0
    by_layer = defaultdict(lambda: {"n": 0, "hits": 0, "pnl": 0.0})
    # 信心分层（2026-08-14：验证"置信度是否真的预示命中/盈亏"——
    # 若高信心段 ROI 明显更好，说明信任门槛有效；若相反则 confidence 是噪声）
    conf_bands = [(0.0, 0.10, "C1 <10%"), (0.10, 0.15, "C2 10-15%"),
                  (0.15, 0.20, "C3 15-20%"), (0.20, 1.01, "C4 ≥20%")]
    by_conf = {tag: {"n": 0, "hits": 0, "pnl": 0.0, "pnl_edge": 0.0, "n_edge": 0}
               for _, _, tag in conf_bands}
    for r in rows:
        probs = r[source_key]
        if probs is None:
            continue
        h, d, a = probs
        sel = max(("home", h), ("draw", d), ("away", a), key=lambda x: x[1])[0]
        odds = r["odds"].get(sel)
        n += 1
        won = sel == r["actual"]
        hit += won
        # Brier（one-hot）
        one_hot = [0.0, 0.0, 0.0]
        one_hot[["home", "draw", "away"].index(r["actual"])] = 1.0
        brier_sum += sum((x - y) ** 2 for x, y in zip(probs, one_hot))
        # ROI（flat 1u）
        if odds and odds > 1.0:
            pnl = (odds - 1.0) if won else -1.0
            roi_always += pnl
            by_layer[layer_of(odds)]["n"] += 1
            by_layer[layer_of(odds)]["hits"] += won
            by_layer[layer_of(odds)]["pnl"] += pnl
            p_sel = {"home": h, "draw": d, "away": a}[sel]
            # 信心分桶（按预测 confidence）
            for lo, hi, tag in conf_bands:
                if lo <= r["conf"] < hi:
                    by_conf[tag]["n"] += 1
                    by_conf[tag]["hits"] += won
                    by_conf[tag]["pnl"] += pnl
                    if p_sel > 1.0 / odds:
                        by_conf[tag]["pnl_edge"] += pnl
                        by_conf[tag]["n_edge"] += 1
                    break
            if p_sel > 1.0 / odds:  # 正期望才押（Kelly 门槛近似）
                roi_edge += pnl
                n_edge += 1
    if n == 0:
        return {"n": 0}
    layers_out = {}
    for k, v in by_layer.items():
        if v["n"]:
            layers_out[k] = {
                "n": v["n"],
                "hit_rate": v["hits"] / v["n"],
                "roi": v["pnl"] / v["n"],
            }
    conf_out = {}
    for _, _, tag in conf_bands:
        v = by_conf[tag]
        if v["n"]:
            conf_out[tag] = {
                "n": v["n"],
                "hit_rate": v["hits"] / v["n"],
                "roi": v["pnl"] / v["n"],
                "roi_edge": (v["pnl_edge"] / v["n_edge"]) if v["n_edge"] else None,
                "n_edge": v["n_edge"],
            }
    return {
        "n": int(n),
        "hit_rate": hit / n,
        "brier": brier_sum / n,
        "roi_always": roi_always / n,
        "roi_edge": (roi_edge / n_edge) if n_edge else None,
        "n_edge": n_edge,
        "by_layer": layers_out,
        "by_confidence": conf_out,
    }


def main() -> dict:
    rows = load_settled()
    print(f"已结算预测: {len(rows)} 场")
    print(f"{'源':8s} {'n':>5s} {'命中率':>8s} {'Brier':>8s} {'ROI全押':>9s} {'ROI仅正EV':>10s}")
    report = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "n_matches": len(rows)}
    labels = {"market": "市场", "model": "模型", "fused": "融合(生产)"}
    for key, label in labels.items():
        s = _stats_for(rows, key)
        if not s.get("n"):
            print(f"{label:8s} 样本不足")
            continue
        report[key] = s
        re_ = f"{s['roi_edge']:+.1%}" if s["roi_edge"] is not None else "  -"
        print(f"{label:8s} {s['n']:5d} {s['hit_rate']:7.1%} {s['brier']:8.4f} "
              f"{s['roi_always']:+8.1%} {re_:>10s}")
    # 系统的真实记录方向（含 R1 改判等生产逻辑）——与"融合 argmax"对比
    sys_hit = sum(1 for r in rows if r["system_direction"] == r["actual"] and r["system_direction"])
    sys_n = sum(1 for r in rows if r["system_direction"])
    if sys_n:
        report["system_actual"] = {"n": sys_n, "hit_rate": sys_hit / sys_n}
        print(f"{'系统实际':8s} {sys_n:5d} {sys_hit/sys_n:7.1%}  (含 R1 改判等生产逻辑)")

    print("\n按赔率层的 ROI(全押 1u)：")
    for key, label in labels.items():
        s = report.get(key, {})
        layers = s.get("by_layer", {})
        if not layers:
            continue
        parts = [f"{k.split(' ')[0]}={v['roi']:+.0%}(n={v['n']})" for k, v in layers.items()]
        print(f"  {label}: {' '.join(parts)}")

    # 信心分层（2026-08-14）：融合源的命中率/ROI 是否随置信度单调？
    # 若高信心段（C4）ROI 显著为正 → 信任门槛有效；若平坦/反转 → confidence 是噪声。
    fused = report.get("fused", {})
    conf = fused.get("by_confidence", {})
    if conf:
        print("\n融合源 · 按置信度分层（验证 confidence 是否预示盈亏）：")
        print(f"  {'段位':10s} {'n':>4s} {'命中率':>8s} {'ROI全押':>9s} {'ROI仅正EV':>10s}")
        for tag, v in conf.items():
            re_ = f"{v['roi_edge']:+.1%}" if v["roi_edge"] is not None else "  -"
            print(f"  {tag:10s} {v['n']:4d} {v['hit_rate']:7.1%} {v['roi']:+8.1%} {re_:>10s}")

    out = ROOT / "data" / "state" / "market_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out}")
    return report


if __name__ == "__main__":
    main()
