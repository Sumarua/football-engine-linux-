"""每日流水线（pipeline）包 — 2026-08-14 拆分

把 engine/main.py 中可独立测试的纯函数/步骤抽到这里，main.py 只做编排。

当前进度（巨型函数拆分的分阶段策略）：
- run_daily_pipeline（~1220 行）与 run_settlement（~843 行）主体仍留在
  main.py——它们是强耦合的顺序流水线（共享 30+ 局部变量），在无集成测试
  兜底前整体搬移风险高。先抽纯单元 + 写回归测试，后续每步验证后再继续拆。
- 已抽出：
  helpers.py    纯工具（load_config / 联赛归一化 / 方向选择 / 分档）
  sina_odds.py  新浪赔率加载（load_sina_odds_map）
  market_signal 盘口信号方向（engine/market_signal.py，由本包模块引用）
  beijing_time  北京时间口径（engine/beijing_time.py）

阶段图（run_daily_pipeline）：
  [1/8] 抓取赛程(三源) → [1.5/8] DJYY增强 → [2/8] 球队评级 → [3/8] 模型初始化
  → [4/8] 预测+增强 → [5/8] 策略/出票 → [6/8] 落盘 → [7/8] 锁定 → [8/8] 推送

阶段图（run_settlement）：
  1)全局预测索引 → 2)幂等键 → 3)赛果归一化 → 3.5)同场合并 → 4)Elo
  → 5)results.json → 6)MatchDB → 7)熔断+逐场结算 → 8)在线权重/组合挖掘
  → 8.5)赛果回写 → 9)CPPI → 10)复盘 → 10.5)高置信反向样本 → 11)重型校准
"""
from __future__ import annotations
