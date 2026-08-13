"""比分/总进球概率校准（2026-08-14 新增）

背景：模型比分矩阵（top_scores/total_goals）系统性偏向低比分——
- 历史 top1 命中仅 13.3%（top5 54.9%），top1 为 ≥3 球的占比 31% vs 实际 50%
- 模型 total_goals 低估 3+ 球（3球 17% vs 实际 22%，4球 10% vs 15%）

做法：贝叶斯收缩——把模型的比分/总进球分布向"历史真实分布"（经验先验）
收缩。经验先验由 results.json 全量赛果统计（810 场），是可靠的无偏基准。
calibrated = (1-shrink) * model + shrink * prior，再归一化。

shrink 取 0.5（模型与历史各半），可通过 backtest 调。经验先验在 build_site
结算后重建（滞后一天，天然无未来泄漏）。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PRIOR_PATH = ROOT / "data" / "state" / "score_prior.json"


def build_score_prior(daily_root: Path | None = None, out_path: Path | None = None) -> dict:
    """扫描全量 results.json，统计经验比分/总进球/半全场分布。"""
    daily_root = daily_root or ROOT / "data" / "daily"
    out_path = out_path or DEFAULT_PRIOR_PATH

    score_cnt: Counter = Counter()
    total_cnt: Counter = Counter()
    htft_cnt: Counter = Counter()
    n = 0

    for rf in sorted(daily_root.glob("*/results.json")):
        try:
            results = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in results:
            hs, as_ = r.get("home_score"), r.get("away_score")
            if hs is None or as_ is None:
                continue
            hs, as_ = int(hs), int(as_)
            score_cnt[(hs, as_)] += 1
            total_cnt[hs + as_] += 1
            if hs > as_:
                htft_cnt["HH"] += 1  # 半全场仅用全场结果近似（htft 模型另校准）
            elif hs == as_:
                htft_cnt["DD"] += 1
            else:
                htft_cnt["AA"] += 1
            n += 1

    def _norm(cnt: Counter) -> dict:
        total = sum(cnt.values())
        if total <= 0:
            return {}
        return {k: round(v / total, 6) for k, v in cnt.items()}

    # 比分键 (hs, as_) 转 "hs-as" 字符串（JSON 可序列化）
    score_norm = {f"{k[0]}-{k[1]}": round(v / sum(score_cnt.values()), 6) for k, v in score_cnt.items()} if score_cnt else {}
    total_norm = {str(k): round(v / sum(total_cnt.values()), 6) for k, v in total_cnt.items()} if total_cnt else {}

    prior = {
        "n_matches": n,
        "score": score_norm,
        "total_goals": total_norm,
        "built_at": "",
    }
    from datetime import datetime
    prior["built_at"] = datetime.now().isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prior, ensure_ascii=False, indent=1))
    return prior


def load_score_prior(path: Path | None = None) -> dict:
    """加载经验先验；缺失时现场重建（避免首次运行无先验）。"""
    path = path or DEFAULT_PRIOR_PATH
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # 现场重建（幂等）
    return build_score_prior(out_path=path)


def calibrate_score_probs(top_scores: list, prior: dict, shrink: float = 0.5) -> list:
    """把模型 top_scores [[h,a,p],...] 向经验比分分布收缩，返回同格式列表（归一化）。"""
    score_prior = (prior or {}).get("score") or {}
    if not top_scores or not score_prior:
        return top_scores

    # 经验先验里没有的比分 → 给一个极小地板（避免完全抹掉模型的长尾比分）
    _floor = 1e-4
    calibrated = []
    for row in top_scores:
        try:
            hs, as_, p = int(row[0]), int(row[1]), float(row[2])
        except (IndexError, TypeError, ValueError):
            calibrated.append(row)
            continue
        q = score_prior.get(f"{hs}-{as_}", _floor)
        calibrated.append([hs, as_, (1 - shrink) * p + shrink * q])

    # 归一化
    total = sum(c[2] for c in calibrated)
    if total > 0:
        calibrated = [[c[0], c[1], c[2] / total] for c in calibrated]
    return calibrated


def calibrate_total_goals(total_goals: list, prior: dict, shrink: float = 0.5) -> list:
    """把模型 total_goals [[g,p],...] 向经验总进球分布收缩，返回同格式列表。"""
    tg_prior = (prior or {}).get("total_goals") or {}
    if not total_goals or not tg_prior:
        return total_goals

    _floor = 1e-4
    calibrated = []
    for row in total_goals:
        try:
            g, p = int(row[0]), float(row[1])
        except (IndexError, TypeError, ValueError):
            calibrated.append(row)
            continue
        q = tg_prior.get(str(g), _floor)
        calibrated.append([g, (1 - shrink) * p + shrink * q])

    total = sum(c[1] for c in calibrated)
    if total > 0:
        calibrated = [[c[0], c[1] / total] for c in calibrated]
    return calibrated


def empirical_score_prob(home: int, away: int, prior: dict, floor: float = 0.001) -> float:
    """某具体比分的经验频率（来自全量赛果），用于比分串单腿命中率校准。

    比全局 top1 命中率（0.122）更细：1-0 ≈ 13.2%、0-0 ≈ 6.7%、3-2 ≈ 2.0%。
    """
    score_prior = (prior or {}).get("score") or {}
    return float(score_prior.get(f"{home}-{away}", floor))
