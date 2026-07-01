# skill-northbound-margin-monitor

A-share 北向资金 + 融资融券全景监控系统。每日自动采集沪深港通资金流向和两融数据，通过 12 个信号检测器和 4 个共振/背离分析引擎生成综合情绪评分日报。

## 功能特性

- **双源容灾**: Pandadata（融资个股）+ 东方财富（北向资金），自动降级切换
- **多信号引擎**: 12 个独立检测器，基于 registry 模式可扩展
- **共振/背离检测**: 4 种跨市场信号模式识别
- **综合评分**: 0-100 量化情绪评分，含风险惩罚机制
- **双格式输出**: Markdown 可读日报 + JSON 结构化数据
- **日期缓存**: 同日数据自动复用，避免重复 API 调用
- **数据溯源**: 报告明确标注每个数据类别的来源和真实性状态

## 安装

```bash
# 克隆仓库
cd D:\python\PandaAI\Pandaai_skill_02

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 Pandadata 凭证:
#   DEFAULT_USERNAME=86xxxxxxxxxxx
#   DEFAULT_PASSWORD=your_password
```

### 依赖

- Python >= 3.10
- panda_data >= 0.0.9 (融资融券数据)
- akshare >= 1.18.0 (北向资金数据)
- pandas >= 2.0.0
- numpy >= 1.22, < 2.0

## 使用

```bash
# 默认运行（最新交易日，使用缓存）
python run.py

# 指定日期
python run.py --date 20260630

# 强制拉取最新数据（跳过缓存）
python run.py --date 20260630 --no-cache

# 自定义 TOP-N 数量
python run.py --top-n 30

# 详细日志
python run.py --verbose

# 清理缓存（删除30天前的数据）
python run.py --cleanup-cache 30
```

## 输出

```
output/
  2026-06-30/
    panorama_monitor_20260630.md   # Markdown 日报
    panorama_monitor_20260630.json # JSON 结构化数据
```

### 日报包含

1. **北向资金概览** — 6 信号检测结果、数据摘要
2. **融资融券概览** — 6 信号检测结果、宏观数据、TOP10 排名
3. **共振/背离分析** — 偏多/偏空共振、谨慎偏多/背离危险
4. **综合情绪评估** — 0-100 评分 + 等级 + 风险惩罚
5. **板块资金流向** — A 股行业分布统计
6. **风险提示** — 汇总所有高风险信号
7. **数据溯源** — 每个数据源的状态标注

## 项目结构

```
core/
  __init__.py
  cache.py          # 日期分区 Parquet 缓存
  data_fetcher.py   # Pandadata + AKShare 双源容灾数据采集
  northbound.py     # 北向资金 6 信号检测器 + registry
  margin.py         # 融资融券 6 信号检测器 + registry
  resonance.py      # 4 共振/背离模式检测
  scorer.py         # 综合评分 0-100 + 排名
  reporter.py       # Markdown + JSON 报告生成
  pipeline.py       # PanoramaPipeline 编排器
run.py              # CLI 入口
config.json         # 阈值和权重配置
tests/              # 152 个测试用例
  conftest.py       # 共享 fixtures
  test_cache.py
  test_northbound.py
  test_margin.py
  test_resonance.py
  test_scorer.py
  test_reporter.py
llm/                # LLM 宏观分析模块
  analyst.py        # DeepSeek/Claude API + 规则引擎回退
```

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定模块
python -m pytest tests/test_northbound.py -v

# 带覆盖率
python -m pytest tests/ --cov=core --cov-report=term-missing
```

## 配置

`config.json` 中可调整所有阈值：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `northbound.consecutive_days_threshold` | 3 | 连续流入/流出天数阈值 |
| `northbound.anomaly_zscore_threshold` | 2.0 | 单日异常 Z-score 阈值 |
| `margin.buy_ratio_hot` | 0.10 | 融资买入比活跃线 |
| `margin.buy_ratio_dangerous` | 0.15 | 融资买入比危险线 |
| `scoring.northbound_weight` | 0.5 | 北向资金在综合评分中权重 |
| `scoring.margin_weight` | 0.5 | 融资融券在综合评分中权重 |

## 数据源

| 数据 | 主源 | 备源 | 状态 |
|------|------|------|------|
| 融资融券个股 | Pandadata `get_margin()` | 东方财富 SSE/SZSE | ✅ 实时 |
| 融资融券宏观 | 东方财富 `macro_china_market_margin_sh/sz` | — | ✅ 实时 |
| 北向资金汇总 | 东方财富 `stock_hsgt_hist_em` | — | ⚠️ 部分字段滞后 |
| 北向资金流向 | 东方财富 `stock_hsgt_fund_flow_summary_em` | — | ✅ 实时 |
| 股票信息 | Pandadata `get_stock_detail` | — | ✅ 实时 |

## License

GPLv3
