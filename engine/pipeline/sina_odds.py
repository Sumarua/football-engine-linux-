"""新浪赔率加载（从 engine/main.py 抽出，2026-08-14）

load_sina_odds_map: 读取 data/daily/{date}/odds_sina.json，建立
(队名) 与 (竞彩编号) 两套索引。文件不存在或解析失败返回空 dict（容错，
与旧行为一致——调用方按 missing 处理即可）。
"""
from __future__ import annotations

import json
from pathlib import Path


def load_sina_odds_map(target_date, root: Path | None = None) -> tuple[dict, dict, int]:
    """加载指定日期的新浪赔率。

    Returns:
        (sina_odds_map, sina_odds_by_no, n_matches):
          sina_odds_map:   {(home_team, away_team): match}（队名索引，fallback）
          sina_odds_by_no: {match_no: match}（竞彩编号索引，优先匹配）
          n_matches:       成功解析的场次数（用于日志；0 = 缺失或解析失败）
    """
    root = root or Path(__file__).resolve().parent.parent.parent
    sina_odds_map: dict = {}
    sina_odds_by_no: dict = {}
    sina_odds_file = root / "data" / "daily" / str(target_date) / "odds_sina.json"
    n_matches = 0
    if sina_odds_file.exists():
        try:
            sina_data = json.loads(sina_odds_file.read_text(encoding="utf-8"))
            n_matches = len(sina_data)
            for m in sina_data:
                # 按队名索引（fallback）
                sina_odds_map[(m.get("home_team", ""), m.get("away_team", ""))] = m
                # 按竞彩编号索引（优先）
                match_no = m.get("match_no", "")
                if match_no:
                    sina_odds_by_no[match_no] = m
        except Exception:
            pass
    return sina_odds_map, sina_odds_by_no, n_matches
