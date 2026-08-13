"""联赛分层价值报告 — 回答"哪个联赛值得投"

原理：
- 聚合所有历史 predictions + 赛果
- 按联赛分层：场数 / 方向命中率 / 平均主推赔率 / 每场押 1 单位 ROI
- 结论：命中率高且赔率高的联赛 = 价值区（值得投）
        命中率低 = 送钱区（避开，模型对该联赛水土不服）
- 与 league_params 自适应联动：联赛积累样本越多，参数越可信

输出 league_report.json 供页面展示与投注决策参考。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# 送钱区/价值区定性所需最小累计样本（2026-08-14：n=10-24 时噪声主导，
# 翻转 verdict 会误禁/漏禁；n≥50 才允许定性，否则只给温和的谨慎/观望）
MIN_VERDICT_SAMPLES = 50


def classify_league(
    n: int, roi: float, hit_rate: float,
    recent5_n: int, recent5_hit_rate: float,
    recent10_n: int, recent10_hit_rate: float,
) -> str:
    """联赛 verdict 判定（2026-08-14 拆出为纯函数，供回归测试锁定 n≥50 规则）。

    规则：
    - n<3：样本不足
    - 3≤n<50：不给"送钱区/价值区"定性（小样本噪声主导），只给谨慎/观望
    - n≥50：按 ROI/命中率定性；送钱区且近5场≥60% 且 近10场≥50% → 回暖解禁
    """
    if n < 3:
        return "样本不足"
    if n < MIN_VERDICT_SAMPLES:
        return "谨慎" if roi < 0 else "观望"
    if roi > 0.05 and hit_rate >= 0.5:
        return "价值区"
    if roi < -0.05 or hit_rate < 0.4:
        # 回暖解禁（2026-08-06，用户需求）：累计口径送钱区，但最近窗口命中率 ≥60%
        # （且 ≥3 场）→ 解禁观察，不再禁投。再拉胯会自动打回送钱区（累计口径兜底）。
        # 2026-08-13 双窗口判定：近5场 ≥60% **且** 近10场 ≥50% 才解禁——
        # 单看5场容易误判（如 3中2=67% 但10场 4中6=40% 其实在恶化）。
        if recent5_n >= 3 and recent5_hit_rate >= 0.6 \
                and recent10_n >= 5 and recent10_hit_rate >= 0.5:
            return "回暖解禁"
        return "送钱区"
    if roi < 0:
        return "谨慎"
    return "观望"


def build_league_report(
    daily_root: Path | None = None,
    out_path: Path | None = None,
    external_samples: list[dict] | None = None,
) -> dict:
    daily_root = daily_root or Path("data/daily")
    out_path = out_path or Path("data/state/league_report.json")

    # 联赛 → 统计
    stats: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "hits": 0, "odds_sum": 0.0, "roi_sum": 0.0,
        "confidence_sum": 0.0, "history": [],  # [(date, hit)] 时间序，供近期窗口判定
    })

    for pf in sorted(daily_root.glob("*/predictions.json")):
        try:
            preds = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue
        _date = pf.parent.name  # YYYY-MM-DD 目录名 = 时间序
        for p in preds:
            lg = p.get("competition") or "未知"
            if p.get("actual_home_score") is None:
                continue  # 未结算
            hs, as_ = p["actual_home_score"], p["actual_away_score"]
            actual = "home" if hs > as_ else ("draw" if hs == as_ else "away")
            direction = p.get("direction")
            if not direction:
                probs = (p.get("home_win_prob", 0), p.get("draw_prob", 0), p.get("away_win_prob", 0))
                direction = ["home", "draw", "away"][probs.index(max(probs))]
            odds = p.get(f"{direction}_odds") or 0
            hit = direction == actual
            s = stats[lg]
            s["n"] += 1
            s["hits"] += 1 if hit else 0
            s["odds_sum"] += odds or 2.0
            s["confidence_sum"] += p.get("confidence", 0)
            # 每场押 1 单位 ROI（含抽水后实际盈亏）
            if odds > 0:
                s["roi_sum"] += (odds - 1) if hit else -1.0
            # 近期窗口（2026-08-06 实现）：按日期目录序收集，末尾取最近 RECENT_WINDOW 场
            s["history"].append((_date, hit, odds))

    # 外部历史样本注入（老系统 world-cup-predictor 的联赛复盘：
    # 世界杯积累的是另一预测域，不能搬参数；但同域联赛样本可合并，
    # 让联赛分层判断更快收敛）
    external_added = 0
    for ex in external_samples or []:
        lg = ex.get("competition") or "未知"
        direction = ex.get("direction")
        actual = ex.get("actual")
        odds = ex.get("odds") or 0
        if not direction or not actual:
            continue
        s = stats[lg]
        s["n"] += 1
        s["hits"] += 1 if direction == actual else 0
        s["odds_sum"] += odds or 2.0
        s["confidence_sum"] += ex.get("confidence", 0.0)
        if odds > 0:
            s["roi_sum"] += (odds - 1) if direction == actual else -1.0
        external_added += 1

    # 组装报告
    rows = []
    for lg, s in stats.items():
        if s["n"] < 1:
            continue
        hit_rate = s["hits"] / s["n"]
        avg_odds = s["odds_sum"] / s["n"]
        roi = s["roi_sum"] / s["n"]

        def _window(n: int) -> dict:
            """截取最近 n 场窗口统计（n=0 表示全部）"""
            w = s["history"][-n:] if n > 0 else s["history"]
            wn = len(w)
            wh = sum(1 for _d, _h, _o in w if _h)
            wroi = sum((_o - 1) if _h else -1.0 for _d, _h, _o in w) / max(wn, 1)
            return {"n": wn, "hits": wh, "hit_rate": wh / wn if wn else 0.0, "roi": wroi}

        # 窗口：近5场（回暖解禁灵敏度）+ 近期10场（趋势）+ 全部（累计兜底）
        # 2026-08-13 用户反馈：不能只看近5场——样本少时单场异常主导，
        # 近5/近10/全部三窗口对照才能科学判断联赛状态。
        recent5 = _window(5)
        recent10 = _window(10)
        # 2026-08-14：verdict 收敛到纯函数 classify_league（n≥50 规则，
        # 小样本只给谨慎/观望，禁投/价值定性需要实打实证据）。
        verdict = classify_league(
            s["n"], roi, hit_rate,
            recent5["n"], recent5["hit_rate"],
            recent10["n"], recent10["hit_rate"],
        )
        rows.append({
            "league": lg,
            "n": s["n"],
            "hit_rate": round(hit_rate, 4),
            "avg_odds": round(avg_odds, 3),
            "roi": round(roi, 4),
            "avg_confidence": round(s["confidence_sum"] / s["n"], 4),
            "verdict": verdict,
            # 近期窗口明细（页面展示"最近5场 x/x"）
            "recent_n": recent5["n"],
            "recent_hits": recent5["hits"],
            "recent_hit_rate": round(recent5["hit_rate"], 4),
            "recent_roi": round(recent5["roi"], 4),
            # 近期10场窗口（2026-08-13 新增：三窗口对照）
            "recent10_n": recent10["n"],
            "recent10_hits": recent10["hits"],
            "recent10_hit_rate": round(recent10["hit_rate"], 4),
            "recent10_roi": round(recent10["roi"], 4),
        })

    rows.sort(key=lambda r: (-r["n"], -r["roi"]))
    report = {
        "n_leagues": len(rows),
        "n_matches": sum(r["n"] for r in rows),
        "external_added": external_added,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "leagues": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    r = build_league_report()
    print(f"联赛分层报告: {r['n_leagues']} 个联赛, {r['n_matches']} 场")
    for row in r["leagues"]:
        print(f"  {row['league']}: {row['n']}场 命中{row['hit_rate']*100:.0f}% "
              f"均赔{row['avg_odds']:.2f} ROI{row['roi']*100:+.1f}% [{row['verdict']}]")
