from __future__ import annotations
"""可靠性曲线与校准评估（Reliability Curve / ECE）

借鉴 Hicruben/world-cup-2026-prediction-model:
- 可靠性曲线："说 70% 就真的发生 70%"
- ECE（Expected Calibration Error）+ MCE（Max Calibration Error）
- 分箱（equal-frequency / equal-width）
- 输出曲线数据供 build_site 画图

本地 walk_forward 已有 brier/rps/log_loss/ece 标量，本模块补充:
  1) 完整分箱可靠性曲线（画图用）
  2) 每个箱的 (预测均值, 实际频率, 样本量)
  3) 分箱自适应（样本不足自动合并）
"""

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CalibrationBin:
    index: int
    bin_center: float      # 预测概率均值
    empirical_freq: float  # 实际发生频率
    count: int             # 样本量
    low: float             # 箱下界（预测概率）
    high: float            # 箱上界

    def to_dict(self) -> dict[str, float]:
        return {
            "bin_center": round(float(self.bin_center), 4),
            "empirical_freq": round(float(self.empirical_freq), 4),
            "count": int(self.count),
            "low": round(float(self.low), 4),
            "high": round(float(self.high), 4),
            "gap": round(float(self.empirical_freq - self.bin_center), 4),
        }


@dataclass
class ReliabilityReport:
    bins: list[CalibrationBin] = field(default_factory=list)
    ece: float = 0.0
    mce: float = 0.0
    n_samples: int = 0
    # 汇总：校准良好的箱占比（|gap| < 0.05）
    well_calibrated_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": round(float(self.ece), 4),
            "mce": round(float(self.mce), 4),
            "n_samples": int(self.n_samples),
            "well_calibrated_ratio": round(float(self.well_calibrated_ratio), 4),
            "bins": [b.to_dict() for b in self.bins],
        }


def _bin_indices(probs: np.ndarray, n_bins: int) -> np.ndarray:
    """等宽分箱。边界处理：1.0 归入最后一箱。"""
    clipped = np.clip(probs, 0.0, 1.0 - 1e-12)
    return np.clip((clipped * n_bins).astype(int), 0, n_bins - 1)


def compute_reliability(
    probs: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
    *,
    min_bin_count: int = 10,
) -> ReliabilityReport:
    """计算可靠性曲线数据 + ECE/MCE。

    参数
    ----
    probs : 预测概率（0~1），例如主胜概率
    actuals : 实际 0/1（1=事件发生）
    n_bins : 目标分箱数
    min_bin_count : 低于此样本量的相邻箱自动合并
    """
    probs = np.asarray(probs, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if probs.shape != actuals.shape or probs.ndim != 1:
        raise ValueError("probs 与 actuals 必须是一维等长数组")
    if len(probs) == 0:
        return ReliabilityReport()

    indices = _bin_indices(probs, n_bins)

    # 逐箱统计
    raw_bins: list[CalibrationBin] = []
    for i in range(n_bins):
        mask = indices == i
        count = int(mask.sum())
        if count == 0:
            continue
        bin_probs = probs[mask]
        bin_actuals = actuals[mask]
        raw_bins.append(
            CalibrationBin(
                index=i,
                bin_center=float(bin_probs.mean()),
                empirical_freq=float(bin_actuals.mean()),
                count=count,
                low=float(bin_probs.min()),
                high=float(bin_probs.max()),
            )
        )

    # 合并小样本相邻箱（从低概率向高概率方向）
    merged: list[CalibrationBin] = []
    for b in raw_bins:
        if merged and merged[-1].count < min_bin_count:
            prev = merged[-1]
            combined_count = prev.count + b.count
            combined_center = (
                prev.bin_center * prev.count + b.bin_center * b.count
            ) / combined_count
            combined_freq = (
                prev.empirical_freq * prev.count + b.empirical_freq * b.count
            ) / combined_count
            merged[-1] = CalibrationBin(
                index=prev.index,
                bin_center=combined_center,
                empirical_freq=combined_freq,
                count=combined_count,
                low=min(prev.low, b.low),
                high=max(prev.high, b.high),
            )
        else:
            merged.append(b)
    # 末尾若仍不足，并入前箱
    if len(merged) >= 2 and merged[-1].count < min_bin_count:
        prev = merged[-2]
        b = merged[-1]
        combined_count = prev.count + b.count
        merged[-2] = CalibrationBin(
            index=prev.index,
            bin_center=(prev.bin_center * prev.count + b.bin_center * b.count)
            / combined_count,
            empirical_freq=(
                prev.empirical_freq * prev.count + b.empirical_freq * b.count
            ) / combined_count,
            count=combined_count,
            low=min(prev.low, b.low),
            high=max(prev.high, b.high),
        )
        merged.pop()

    # ECE / MCE（按样本量加权）
    total = int(sum(b.count for b in merged))
    ece = sum(
        b.count * abs(b.empirical_freq - b.bin_center) for b in merged
    ) / total if total else 0.0
    mce = max(
        (abs(b.empirical_freq - b.bin_center) for b in merged), default=0.0
    )
    well = sum(
        1 for b in merged if abs(b.empirical_freq - b.bin_center) < 0.05
    )

    return ReliabilityReport(
        bins=merged,
        ece=float(ece),
        mce=float(mce),
        n_samples=total,
        well_calibrated_ratio=well / len(merged) if merged else 0.0,
    )


def compute_outcome_reliability(
    prob_matrix: np.ndarray,
    actual_indices: np.ndarray,
    n_bins: int = 10,
) -> dict[str, ReliabilityReport]:
    """对三分概率矩阵 (N,3) 逐类计算可靠性。

    返回 {"home": report, "draw": report, "away": report}
    """
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    actual_indices = np.asarray(actual_indices, dtype=int)
    labels = ("home", "draw", "away")
    reports: dict[str, ReliabilityReport] = {}
    for col, label in enumerate(labels):
        probs = prob_matrix[:, col]
        actuals = (actual_indices == col).astype(float)
        reports[label] = compute_reliability(probs, actuals, n_bins)
    return reports


def save_report(report: ReliabilityReport, path) -> None:
    """保存可靠性报告 JSON（供 build_site 渲染）。"""
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    # 自检：完美校准的模型 ECE≈0
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.05, 0.95, 2000)
    actuals = rng.binomial(1, probs)
    rep = compute_reliability(probs, actuals)
    print("perfect model ECE:", round(rep.ece, 4), "MCE:", round(rep.mce, 4))
    print("bins:", len(rep.bins))
    print(rep.to_dict())
