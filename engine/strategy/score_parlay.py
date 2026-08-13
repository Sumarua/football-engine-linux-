"""
比分串（波胆过关）生成器 — 竞彩比分玩法 2串1/3串1/3串4容错。

2026-08-08 新增：用户"比分的串也可以搞一个"。竞彩比分赔率高（6-10 倍），
比分 2串1 常见赔率 40-80 倍，是"彩票票"（10% 小注搏大奖）的天然载体。
2026-08-10 扩展：用户"二串三串都应该有，三串打2串1容错率高些" →
  3串1（全中 686 倍）+ 3串4容错（3场出4注：3×2串1+1×3串1，错1场仍中1注2串1）。

数据现实（必须诚实标注）：
- top_scores 来自 DJYY 第三方（score_sources="djyy"），概率未校准，
  且 0-0 系统性高估（8/8 六场 top1=0-0，最高 42.4% vs 实际 0-0 频率 ~8-10%）。
  模块内做 0-0 封顶修正（cap 15%，超额分摊给其他比分）。
- 官方波胆赔率（crs_odds）大部分日期未抓到 → 无官方赔率时用
  竞彩波胆基准赔率表模拟（1-0/0-1≈7倍、2-1≈9.5倍…），标注模拟赔率。
- 比分命中率极低（top1 约 10-13%），串起来期望几乎必然为负 →
  定位娱乐串，注额 ¥2/注，不推荐重注。

设计：
- 胆材：每场修正后 top1 比分，概率 ≥ min_prob(0.12)（太低没串的意义）
- 串法：2串1 top3 组合 + 3串1 top3 + 3串4容错 top3（4注）
- 概率 = 各腿修正概率连乘（模型口径，未校准，页面标注）
- 赔率 = 官方 crs_odds 优先，否则基准表模拟
- 注额 = ¥2/注（彩票票定位）；3串4 = 4注 ¥8
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

STAKE_UNIT = 2.0

# DJYY top1 比分精确命中率（账本 197 场实测：24/197 = 12.2%，2026-08-13）
# 用实测校准替代 DJYY 高估概率——比分串 EV 全靠它把关。
SCORE_CAL_PROB = 0.122

# 0-0 概率封顶（DJYY 系统性高估；实际 0-0 频率 ~8-10%）
ZZ_CAP = 0.15

# 竞彩波胆基准赔率（无官方赔率时模拟用；经验区间：最常见比分赔率最低）
# 2026-08-08：概率反推（1/p×0.55）失真——DJYY top1 概率是市场的 2-3 倍，
# 反推赔率低到 1.4x，比分串失去"高赔率搏大奖"的意义。改用基准赔率表。
BASE_ODDS = {
    (1, 0): 7.0, (0, 1): 7.0, (1, 1): 7.5,
    (2, 0): 10.0, (0, 2): 10.0, (2, 1): 9.5, (1, 2): 9.5,
    (0, 0): 9.0, (2, 2): 14.0,
    (3, 0): 18.0, (0, 3): 18.0, (3, 1): 17.0, (1, 3): 17.0,
    (3, 2): 25.0, (2, 3): 25.0, (3, 3): 30.0,
    (4, 0): 30.0, (0, 4): 30.0, (4, 1): 30.0, (1, 4): 30.0,
}


@dataclass
class ScoreLeg:
    match_id: str
    home_team: str
    away_team: str
    competition: str
    score: tuple[int, int]  # (主, 客)
    prob: float             # 修正后模型概率（DJYY 口径）
    odds: float             # 官方波胆赔率（无则模拟）
    cal_prob: float = 0.0   # 校准后命中率（2026-08-13：DJYY 系统性高估 → prob^1.5 封顶 15%）
    odds_source: str = "official"  # "official" / "simulated"（2026-08-13 起只出 official）


@dataclass
class ScoreTicket:
    parlay_type: str        # "比分2串1" / "比分3串1" / "比分3串4(容错)"
    legs: list[ScoreLeg] = field(default_factory=list)
    total_odds: float = 0.0
    hit_prob: float = 0.0   # 模型概率连乘（未校准）
    stake: float = 2.0
    potential: float = 0.0  # 全中最高回报
    worst_win: float = 0.0  # 容错最差命中回报（3串4=错1场中1注2串1；2串1/3串1=0）
    n_bets: int = 1         # 注数（3串4=4注）
    odds_source: str = "simulated"  # "official" / "simulated"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.parlay_type,
            "legs": [{
                "match": l.match_id, "home": l.home_team, "away": l.away_team,
                "league": l.competition,
                "score": f"{l.score[0]}-{l.score[1]}",
                "prob": round(l.prob, 3),
                "cal_prob": round(l.cal_prob, 4),
                "odds": round(l.odds, 2),
            } for l in self.legs],
            "total_odds": round(self.total_odds, 2),
            "hit_prob": round(self.hit_prob, 4),
            "stake": round(self.stake, 2),
            "potential": round(self.potential, 2),
            "worst_win": round(self.worst_win, 2),
            "n_bets": self.n_bets,
            "odds_source": self.odds_source,
            "note": self.note,
        }


def _fix_zz_overestimate(scores: list) -> list:
    """0-0 封顶修正：DJYY 0-0 系统性高估（最高 42% vs 实际 ~8-10%）。

    0-0 概率 cap 到 ZZ_CAP，超额按比例分摊给其他比分。
    """
    if not scores:
        return []
    total = sum(x[2] for x in scores) or 1.0
    zz = next((x[2] for x in scores if x[0] == 0 and x[1] == 0), 0.0)
    zz_capped = min(zz, ZZ_CAP)
    scale = (total - zz + zz_capped) / total  # >1：其他比分概率整体上调
    out = []
    for h, a, p in scores:
        if (h, a) == (0, 0):
            out.append((h, a, zz_capped / scale))
        else:
            out.append((h, a, p / scale))
    return out


def _official_odds(pred: dict, score: tuple[int, int]) -> float | None:
    """官方波胆赔率（crs_odds 兼容解析），无则 None。"""
    raw = pred.get("crs_odds") or {}
    if not raw:
        return None
    try:
        from engine.strategy.multi_play_ev import parse_crs_odds
        parsed = parse_crs_odds(raw)
        return parsed.get(score)
    except Exception:
        return None


class ScoreParlayBuilder:
    def __init__(
        self,
        min_prob: float = 0.12,
        max_tickets: int = 6,
        max_legs: int = 3,
    ):
        self.min_prob = min_prob
        self.max_tickets = max_tickets
        self.max_legs = max_legs

    def build(self, predictions: list[dict]) -> list[ScoreTicket]:
        """从 predictions 生成比分串。返回空列表 = 今日无比分串（或全负 EV）。

        2026-08-13 科学性改造（84 张 0 中 ROI -100% 教训）：
        - 只用官方真实波胆赔率（crs_odds），无官方赔率 = 无法验证 EV → 淘汰
          （模拟赔率表算 EV 是自欺欺人，之前 2串1 赔率 49-70 倍纯属纸面）
        - DJYY top_scores 概率未校准（系统性高估 2-3 倍）→ 用账本实测比分
          命中率回退校准（top1 ~12%，整体精确比分 ~10-14%）
        - EV 门槛：校准概率连乘 × 官方赔率 > 1 才出票（正 EV 纪律，
          与单关/让球同款：没边际不出注）
        """
        pool: list[ScoreLeg] = []
        for p in predictions or []:
            ts = p.get("top_scores") or []
            if not ts:
                continue
            fixed = _fix_zz_overestimate([(int(x[0]), int(x[1]), float(x[2])) for x in ts])
            if not fixed:
                continue
            h, a, prob = max(fixed, key=lambda x: x[2])
            if prob < self.min_prob:
                continue
            odds = _official_odds(p, (h, a))
            if odds is None:
                # 无官方赔率 → 淘汰（无法验证 EV，模拟赔率是纸面财富）
                continue
            # 校准：用该具体比分的经验频率（全量赛果），比全局 top1 12.2% 更细。
            # 1-0≈13.2%、0-0≈6.7%、3-2≈2.0%——低赔率波胆 = 送钱，宁可空仓。
            _cal_prob = SCORE_CAL_PROB
            try:
                from engine.learning.score_calibration import load_score_prior, empirical_score_prob
                _cal_prob = empirical_score_prob(h, a, load_score_prior())
            except Exception:
                pass
            cal_prob = _cal_prob
            # 单腿价值门槛：校准概率 × 官方赔率 ≥ 1.05（EV>5%）才入池
            if cal_prob * odds < 1.05:
                continue
            pool.append(ScoreLeg(
                match_id=p.get("match_id", ""),
                home_team=p.get("home_team", ""),
                away_team=p.get("away_team", ""),
                competition=p.get("competition", ""),
                score=(h, a), prob=prob, odds=odds,
            ))
            pool[-1].cal_prob = cal_prob
            pool[-1].odds_source = "official"

        if not pool:
            return []

        tickets: list[ScoreTicket] = []
        pool.sort(key=lambda l: l.cal_prob, reverse=True)

        def _ev(legs: list[ScoreLeg]) -> float:
            """校准概率连乘 × 官方赔率 - 1（>0 才出）"""
            p = 1.0
            for lg in legs:
                p *= lg.cal_prob
            odds = 1.0
            for lg in legs:
                odds *= lg.odds
            return p * odds - 1.0

        # 2串1：概率最高的场次两两组合（只留 top3 正 EV）
        top2 = pool[:6]
        _two_in_one = []
        for l1, l2 in combinations(top2, 2):
            hit = l1.cal_prob * l2.cal_prob
            odds = l1.odds * l2.odds
            if hit * odds <= 1.0:
                continue  # 负 EV 不出
            _two_in_one.append(ScoreTicket(
                parlay_type="比分2串1",
                legs=[l1, l2], total_odds=round(odds, 2),
                hit_prob=hit, stake=STAKE_UNIT, potential=round(STAKE_UNIT * odds, 2),
                odds_source="official",
                note=f"两场比分都中才赢（校准命中 {hit:.1%} × 赔率 {odds:.0f} 倍 = EV {hit*odds-1:+.0%}）",
            ))
        _two_in_one.sort(key=lambda t: t.hit_prob * t.total_odds, reverse=True)
        tickets.extend(_two_in_one[:3])

        # 3串1：top3 场次全中才赢（正 EV 才出）
        if len(pool) >= 3:
            l1, l2, l3 = pool[:3]
            hit = l1.cal_prob * l2.cal_prob * l3.cal_prob
            odds = l1.odds * l2.odds * l3.odds
            if hit * odds > 1.0:
                tickets.append(ScoreTicket(
                    parlay_type="比分3串1",
                    legs=[l1, l2, l3], total_odds=round(odds, 2),
                    hit_prob=hit, stake=STAKE_UNIT, potential=round(STAKE_UNIT * odds, 2),
                    odds_source="official",
                    note=f"三场比分全中才赢（赔率 {odds:.0f} 倍，校准命中 {hit:.2%}）",
                ))

        # 3串4 容错：3 场出 4 注（3×2串1 + 1×3串1），错 1 场仍中 1 注 2串1
        if len(pool) >= 3:
            l1, l2, l3 = pool[:3]
            o2_list = sorted([l1.odds * l2.odds, l2.odds * l3.odds, l1.odds * l3.odds], reverse=True)
            o3 = l1.odds * l2.odds * l3.odds
            n_bets = 4
            stake = STAKE_UNIT * n_bets
            worst = STAKE_UNIT * max(o2_list)
            potential = STAKE_UNIT * (sum(o2_list) + o3)
            # 容错票期望：错1场中1注2串1（3种错法）+ 全中
            p_all = l1.cal_prob * l2.cal_prob * l3.cal_prob
            p_one_miss = (1 - l1.cal_prob) * l2.cal_prob * l3.cal_prob \
                + l1.cal_prob * (1 - l2.cal_prob) * l3.cal_prob \
                + l1.cal_prob * l2.cal_prob * (1 - l3.cal_prob)
            exp_return = p_all * potential + p_one_miss * worst
            if exp_return > stake:
                tickets.append(ScoreTicket(
                    parlay_type="比分3串4(容错)",
                    legs=[l1, l2, l3], total_odds=round(o3, 2),
                    hit_prob=p_all, stake=round(stake, 2),
                    potential=round(potential, 2), worst_win=round(worst, 2),
                    n_bets=n_bets, odds_source="official",
                    note=f"3场出4注（3×2串1+1×3串1）错1场仍中1注2串1≈¥{worst:.0f}，全中≈¥{potential:.0f}",
                ))

        # 排序：2串1 → 3串1 → 3串4(容错) 分组展示，同组按 EV 降序
        order = {"比分2串1": 0, "比分3串1": 1, "比分3串4(容错)": 2}
        tickets.sort(key=lambda t: (order.get(t.parlay_type, 9), -(t.hit_prob * t.total_odds)))
        return tickets[: self.max_tickets]


if __name__ == "__main__":
    import sys
    day = sys.argv[1] if len(sys.argv) > 1 else "2026-08-08"
    preds = json.load(open(f"data/daily/{day}/predictions.json"))
    if isinstance(preds, dict):
        preds = preds.get("predictions", [])
    b = ScoreParlayBuilder()
    plan = b.build(preds)
    print(f"{day}: 比分串 {len(plan)} 张")
    for t in plan:
        d = t.to_dict()
        legs = " + ".join(f"{l['home'][:5]}({l['score']})@{l['odds']:.2f}" for l in d["legs"])
        print(f"  [{d['type']}] {d['odds_source']} 概率{d['hit_prob']:.2%} 投入{d['stake']}元 最高{d['potential']}元 | {legs}")
