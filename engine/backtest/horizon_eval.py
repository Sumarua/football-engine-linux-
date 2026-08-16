from __future__ import annotations
"""预测时点分桶评估（Forecast Horizon Bucketing）

借鉴 JetQiao/football-prediction-skill 的 horizons.py：
- 预测截点（as_of）离开赛时间越近，信息越全，但价值越低（赔率已收敛）
- 离开赛越远，信息越少，但价值越高
- 按 T-24h / T-6h / T-90m / 收盘 分桶评估，回答：
  "我该在什么时点出手？哪个时点的预测最准、EV 最高？"

本地系统每天 11:15 定时跑（开售即跑），可产生多个时点快照。
本模块:
  1) infer_horizon(as_of, kickoff) → 分桶
  2) 按桶聚合 brier/rps/log_loss/命中率/EV
  3) 输出对比报告（哪个时点最值得下注）
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

BEIJING_TZ = timezone(timedelta(hours=8))

from .ts_split import brier_score, ranked_probability_score, log_loss

# 与 JetQiao 一致的分桶语义
HORIZON_T24H = "t24h"      # ≥ 15 小时
HORIZON_T6H = "t6h"        # ≥ 225 分钟
HORIZON_T90M = "t90m"      # ≥ 45 分钟
HORIZON_CLOSING = "closing"  # < 45 分钟

HORIZON_LABELS = {
    HORIZON_T24H: "开赛前24小时",
    HORIZON_T6H: "开赛前6小时",
    HORIZON_T90M: "开赛前90分钟",
    HORIZON_CLOSING: "收盘市场",
}

HORIZON_ORDER = [HORIZON_T24H, HORIZON_T6H, HORIZON_T90M, HORIZON_CLOSING]


def _parse_ts(value: str) -> datetime:
    """解析 ISO 时间戳（容忍 Z 后缀与无时区）。

    竞彩开球/预测时间均为北京时间；无时区字符串按北京时间处理，
    避免 aware 与 naive datetime 直接相减报错。
    """
    v = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BEIJING_TZ)
    return dt


def infer_horizon(as_of: str, kickoff: str) -> str:
    """按离开赛分钟数分桶。as_of 必须早于 kickoff。"""
    cutoff = _parse_ts(as_of)
    start = _parse_ts(kickoff)
    minutes = (start - cutoff).total_seconds() / 60.0
    if minutes <= 0:
        raise ValueError(f"预测截点必须早于开赛时间: as_of={as_of} kickoff={kickoff}")
    if minutes >= 15 * 60:
        return HORIZON_T24H
    if minutes >= 225:
        return HORIZON_T6H
    if minutes >= 45:
        return HORIZON_T90M
    return HORIZON_CLOSING


@dataclass
class HorizonBucketResult:
    horizon: str
    label: str
    n: int = 0
    brier: float = 0.0
    rps: float = 0.0
    log_loss: float = 0.0
    hit_rate: float = 0.0
    avg_ev: float = 0.0
    total_ev: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "label": self.label,
            "n": self.n,
            "brier": round(self.brier, 4),
            "rps": round(self.rps, 4),
            "log_loss": round(self.log_loss, 4),
            "hit_rate": round(self.hit_rate, 4),
            "avg_ev": round(self.avg_ev, 4),
            "total_ev": round(self.total_ev, 4),
        }


@dataclass
class HorizonReport:
    results: list[HorizonBucketResult] = field(default_factory=list)
    best_horizon: str = ""
    best_metric: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_horizon": self.best_horizon,
            "best_metric": self.best_metric,
            "buckets": [r.to_dict() for r in self.results],
        }


def evaluate_by_horizon(
    records: list[dict[str, Any]],
    *,
    as_of_key: str = "as_of",
    kickoff_key: str = "kickoff",
    prob_key: str = "probs",
    actual_key: str = "actual_idx",
    ev_key: str = "ev",
) -> HorizonReport:
    """按预测时点分桶评估一组预测记录。

    records 每条: {
        "as_of": "2026-08-11T11:15:00+08:00",
        "kickoff": "2026-08-11T19:35:00+08:00",
        "probs": [0.45, 0.30, 0.25],
        "actual_idx": 0,       # 0=主胜 1=平 2=客胜
        "ev": 0.08,            # 该注期望价值（可选）
    }
    """
    buckets: dict[str, list[dict]] = {h: [] for h in HORIZON_ORDER}
    skipped = 0
    for rec in records:
        try:
            h = infer_horizon(rec[as_of_key], rec[kickoff_key])
        except (KeyError, ValueError):
            skipped += 1
            continue
        buckets[h].append(rec)

    results: list[HorizonBucketResult] = []
    for h in HORIZON_ORDER:
        items = buckets[h]
        if not items:
            results.append(HorizonBucketResult(horizon=h, label=HORIZON_LABELS[h]))
            continue
        probs = np.asarray([r[prob_key] for r in items], dtype=float)
        actuals = np.asarray([r[actual_key] for r in items], dtype=int)
        hit = (np.argmax(probs, axis=1) == actuals).mean()
        evs = [r.get(ev_key, 0.0) for r in items if r.get(ev_key) is not None]
        results.append(
            HorizonBucketResult(
                horizon=h,
                label=HORIZON_LABELS[h],
                n=len(items),
                brier=float(brier_score(probs, actuals)),
                rps=float(ranked_probability_score(probs, actuals)),
                log_loss=float(log_loss(probs, actuals)),
                hit_rate=float(hit),
                avg_ev=float(np.mean(evs)) if evs else 0.0,
                total_ev=float(np.sum(evs)) if evs else 0.0,
            )
        )

    report = HorizonReport(results=results)
    # 选最佳桶：brier 最低 + 样本数≥30
    valid = [r for r in results if r.n >= 30]
    if valid:
        best = min(valid, key=lambda r: r.brier)
        report.best_horizon = best.horizon
        report.best_metric = "brier"
    return report


def save_report(report: HorizonReport, path) -> None:
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    # 自检
    print(infer_horizon("2026-08-11T11:15:00+08:00", "2026-08-12T03:00:00+08:00"))  # t24h
    print(infer_horizon("2026-08-11T20:00:00+08:00", "2026-08-11T23:00:00+08:00"))  # t90m?
    print(infer_horizon("2026-08-11T22:30:00+08:00", "2026-08-11T23:00:00+08:00"))  # closing
