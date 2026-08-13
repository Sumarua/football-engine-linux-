"""流水线纯工具函数（从 engine/main.py 抽出，2026-08-14）

全部为无副作用纯函数，供回归测试直接覆盖；main.py 通过
`from engine.pipeline.helpers import ...` 引用，保持原有模块级名字可用
（外部 `from engine.main import load_config` 不受影响）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(name: str) -> dict:
    path = ROOT / "config" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# 联赛名别名归一化（2026-08-12 修复）：新浪/竞彩接口联赛名不稳定，
# 同一联赛可能以多个名字出现（账本实证：瑞超=瑞典超 9场、韩职=K1联赛 3场）。
# 若不做归一化：①R1 平局改判/锚定在高平联赛上静默失效（联赛名不匹配）
# ②账本平局率统计被拆分稀释（瑞典超 14场43% 被拆成 瑞超 9场11%）。
LEAGUE_ALIASES = {"瑞超": "瑞典超", "韩职": "K1联赛"}


def _canon_league(lg: str | None) -> str:
    """联赛名归一化：别名 → 标准名（用于 R1/锚定/联赛反馈/账本分桶）"""
    if not lg:
        return ""
    return LEAGUE_ALIASES.get(lg, lg)


def _pick_direction(h: float, d: float, a: float, draw_alert=None) -> str:
    """从最终概率选方向（argmax），R1 触发时改判平局。

    2026-08-05 已验证：市场平局改判（market_d≥0.30 且 d≥0.22）回测 112 场
    命中率 43.8%→42.9% 不升反降 → 维持原逻辑，勿再盲目调参（walk_forward 二次验证）。

    2026-08-12 数据驱动新证据（非盲目调参）：
    R1 = 高平联赛 + 市场平局P∈[0.20,0.30) 无脑改判平局，回测 137 场
    命中 46.7% vs 基线 43.8%（+2.9pp），切半验证 42.6%/50.7% 均优于各自基线。

    2026-08-13 停用 balanced_draw/cold_draw 改判（实盘证伪）：
    账本 197 场中已改判 13 场仅 3 场改对（23%），10 场把本来正确的
    argmax 改错（净 -3）；改判平局概率均值 0.30 ≈ 实际平局率 29%，
    无信息增益。纯 argmax 命中率 44.2% > 改判后 42.6%（+1.5pp）。
    只保留 R1（league_draw，有独立回测支撑），balanced/cold 仅作
    展示标记不再触发方向改判。
    """
    if draw_alert == "league_draw":
        return "draw"
    return max(("home", h), ("draw", d), ("away", a), key=lambda x: x[1])[0]


def _prob_band(prob: float) -> str:
    """概率分档"""
    if prob >= 0.65:
        return "high"
    elif prob >= 0.45:
        return "mid"
    else:
        return "low"


def _odds_band(odds: float) -> str:
    """赔率分档"""
    if odds < 1.5:
        return "1.0-1.5"
    elif odds < 2.0:
        return "1.5-2.0"
    elif odds < 3.0:
        return "2.0-3.0"
    elif odds < 5.0:
        return "3.0-5.0"
    else:
        return "5.0+"


def _extract_features(fixture, pred) -> dict:
    """从比赛和预测中提取离散特征（用于组合挖掘）"""
    features = {
        "league": fixture.competition or "unknown",
        "prob_band": _prob_band(max(pred.home_win_prob, pred.draw_prob, pred.away_win_prob)),
    }
    if fixture.home_odds:
        features["odds_band"] = _odds_band(fixture.home_odds)
    if fixture.handicap is not None:
        features["handicap"] = str(fixture.handicap)
    return features
