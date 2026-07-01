---
name: skill-northbound-margin-monitor
description: A-share 北向资金+融资融券全景监控，每日生成多信号分析日报
version: 1.0.0
category: quant-skills
triggers:
  - 北向资金
  - 融资融券
  - 全景监控
  - 资金流向
  - margin monitor
data_sources:
  - pandadata (融资融券个股 + 股票信息)
  - eastmoney via akshare (北向资金汇总 + 市场流向)
output_formats:
  - markdown日报
  - json结构化数据
schedule: 每日收盘后
---

# 北向资金 + 融资融券全景监控

## 概述

每日监控 A 股北向资金（沪深港通）和融资融券全景数据，通过 **12 个信号检测器 + 4 个共振/背离模式** 生成综合情绪评分（0-100）和 Markdown/JSON 双格式日报。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 Pandadata 凭证（融资数据必需）
cp .env.example .env
# 编辑 .env 填入 DEFAULT_USERNAME 和 DEFAULT_PASSWORD

# 运行（最新交易日）
python run.py

# 指定日期 + 强制刷新
python run.py --date 20260630 --no-cache

# 调整 TOP-N 数量 + 详细日志
python run.py --top-n 30 --verbose
```

## 核心架构

```
run.py (CLI)
  └─ PanoramaPipeline.run()
       ├─ DataFetcher: Pandadata(融资+股票) + AKShare(北向)
       ├─ Northbound: 6 信号检测器 (registry pattern)
       ├─ Margin:     6 信号检测器 (registry pattern)
       ├─ Resonance:  4 共振/背离模式
       ├─ Scorer:     综合评分 0-100
       └─ Reporter:   Markdown + JSON 双输出
```

## 信号检测器

### 北向资金 (6 signals)
| 检测器 | 权重 | 说明 |
|--------|------|------|
| 净流向趋势 | 3 | 连续N日净流入/流出 |
| 累计趋势 | 3 | 累计净买入 MA20 vs MA60 |
| 单日异常 | 2 | 当日 Z-score vs 60天分布 |
| 板块偏好 | 2 | 行业持仓市值变化 |
| 市场流向 | 2 | 沪股通/深股通方向一致性 |
| 重仓股变动 | 1 | 持仓市值 5/20 日变动 |

### 融资融券 (6 signals)
| 检测器 | 权重 | 说明 |
|--------|------|------|
| 融资余额趋势 | 3 | 当前 vs MA5/MA20，热度评级 |
| 融资买入比 | 3 | 融资买入/总成交，>10%热 >15%危险 |
| 融券趋势 | 2 | 5日融券余额变化率 |
| 融资融券比 | 2 | 融资/融券比率趋势 |
| 融资重仓股 | 1 | TOP20 集中度分析 |
| 融资类型分布 | 1 | 现金/股票担保品分布 |

### 共振/背离 (4 patterns)
| 模式 | 条件 | 信号 |
|------|------|------|
| 偏多共振 | 北向流入 + 融资扩张 | 强烈看多 |
| 偏空共振 | 北向流出 + 融资收缩 | 强烈看空 |
| 谨慎偏多 | 北向流入 + 融资去杠杆 | 聪明钱进场 |
| 背离危险 | 北向流出 + 融资加杠杆 | 外资撤离/散户接盘 |

## 输出

### 日报 (7 部分)
1. 北向资金概览
2. 融资融券概览
3. 共振/背离分析
4. 综合情绪评估 (0-100 评分)
5. 板块资金流向
6. 风险提示
7. 数据溯源

### 文件路径
```
output/YYYY-MM-DD/
  panorama_monitor_YYYYMMDD.md
  panorama_monitor_YYYYMMDD.json
```

## 配置

编辑 `config.json` 调整阈值：

```json
{
  "northbound": {
    "consecutive_days_threshold": 3,
    "anomaly_zscore_threshold": 2.0,
    "ma_short_window": 20,
    "ma_long_window": 60
  },
  "margin": {
    "buy_ratio_hot": 0.10,
    "buy_ratio_dangerous": 0.15,
    "trend_short_window": 5,
    "trend_long_window": 20
  },
  "scoring": {
    "northbound_weight": 0.5,
    "margin_weight": 0.5
  }
}
```

## 数据源说明

| 数据类别 | 主源 | 备源 |
|----------|------|------|
| 融资融券个股 | Pandadata `get_margin()` | 东方财富 `stock_margin_detail_sse/szse` |
| 融资融券宏观 | 东方财富 `macro_china_market_margin_sh/sz` | — |
| 北向资金汇总 | 东方财富 `stock_hsgt_hist_em` | — |
| 北向资金流向 | 东方财富 `stock_hsgt_fund_flow_summary_em` | — |
| 股票信息 | Pandadata `get_stock_detail` | — |

**已知限制**：
- 北向资金 `net_buy_amount` 近2周为 NaN，`market_value` 近3个月为 0（API 限制）
- 检测器自动降级使用 CSI300 指数变动作为替代指标
- 个股北向持仓数据不可直接获取，板块偏好基于汇总数据推算

## 评分等级

| 分数 | 等级 | 含义 |
|------|------|------|
| 80-100 | A | 强烈看多 |
| 65-79 | B | 偏多 |
| 45-64 | C | 中性 |
| 30-44 | D | 中性偏空 |
| 0-29 | E-F | 看空/强烈看空 |
