# 数据源参考手册

## 概览

| 数据类别 | API | 库 | 频率 | 覆盖范围 | 可靠性 |
|----------|-----|-----|------|----------|--------|
| 融资融券个股 | `pdd.get_margin()` | panda_data | 日 | ~5186只A股 | ✅ 高 |
| 融资融券宏观SH | `ak.macro_china_market_margin_sh()` | akshare | 日 | 沪市汇总 | ✅ 高 |
| 融资融券宏观SZ | `ak.macro_china_market_margin_sz()` | akshare | 日 | 深市汇总 | ✅ 高 |
| 融资融券个股(备) | `ak.stock_margin_detail_sse/szse()` | akshare | 日 | ~4000只 | ⚠️ 备源 |
| 股票交易列表 | `pdd.get_trade_list()` | panda_data | 日 | 全A股 | ✅ 高 |
| 股票详情 | `pdd.get_stock_detail()` | panda_data | 日 | 全A股 | ✅ 高 |
| 北向资金汇总 | `ak.stock_hsgt_hist_em()` | akshare | 日 | 沪深港通 | ⚠️ 部分字段缺失 |
| 北向资金流向 | `ak.stock_hsgt_fund_flow_summary_em()` | akshare | 日 | 沪/深股通 | ✅ 高 |

---

## 1. Pandadata — 融资融券

### get_margin

```python
import panda_data as pdd

pdd.init_token(username="86...", password="...", base_url="...")
df = pdd.get_margin(
    symbol=["000001.SZ", "600519.SH"],
    start_date="20260601",
    end_date="20260630",
)
```

**返回字段:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `symbol` | str | 股票代码 (000001.SZ) |
| `date` | str | 日期 (20260630) |
| `margin_balance` | float | 融资余额（元） |
| `buy_on_margin_value` | float | 融资买入额（元） |
| `margin_repayment` | float | 融资偿还额（元） |
| `short_balance` | float | 融券余额（元） |
| `short_sell_quantity` | float | 融券卖出量（股） |
| `short_repayment_quantity` | float | 融券偿还量（股） |
| `short_balance_quantity` | float | 融券余量（股） |
| `total_balance` | float | 总余额（元） |
| `margin_type` | str | 融资类型（现金/股票） |

**批量查询限制:**
- 单次最多 200 只股票
- 建议 4 线程并发
- 5186 只全量数据约 5 分钟
- 返回约 550K 行

### get_trade_list

```python
df = pdd.get_trade_list(date="20260630")
```

返回当日可交易股票列表，包含 `symbol`, `name` 等字段。

### get_stock_detail

```python
df = pdd.get_stock_detail(symbol=["000001.SZ", ...])
```

**关键字段:**
- `symbol` — 股票代码
- `sector_code_name` — 行业分类（映射为 `industry`）
- `status` — 上市状态（映射为 `list_status`）

---

## 2. 东方财富 (via AKShare) — 北向资金

### stock_hsgt_hist_em

```python
import akshare as ak
df = ak.stock_hsgt_hist_em()
```

**原始字段（中文）** → 映射后（英文）:

| 原始字段 | 映射字段 | 数据质量 |
|----------|----------|----------|
| 日期 | `date` | ✅ |
| 当日成交净买额 | `net_buy_amount` | ❌ 近2周 NaN |
| 买入成交额 | `buy_turnover` | ✅ |
| 卖出成交额 | `sell_turnover` | ✅ |
| 历史累计净买额 | `cumulative_net_buy` | ✅ |
| 当日资金流入 | `daily_inflow` | ✅ |
| 当日余额 | `daily_balance` | ✅ |
| 持股市值 | `market_value` | ❌ 2026-04后全为0 |
| 沪深300 | `csi300` | ✅ 用作替代指标 |
| 沪深300-涨跌幅 | `csi300_change` | ✅ |
| 领涨股 | `top_gainer_name` | ✅ |
| 领涨股-涨跌幅 | `top_gainer_pct` | ✅ |
| 领涨股-代码 | `top_gainer_code` | ✅ |

**数据量**: 约 2699 行（2014年至今每个交易日）

### stock_hsgt_fund_flow_summary_em

```python
df = ak.stock_hsgt_fund_flow_summary_em()
```

**返回 4 行**（当前交易日）:

| 板块 | 资金方向 | 关键字段 |
|------|----------|----------|
| 沪股通 | 北向/南向 | 成交净买额, 资金净流入 |
| 港股通(沪) | 北向/南向 | 上涨数, 下跌数 |
| 深股通 | 北向/南向 | 相关指数, 指数涨跌幅 |
| 港股通(深) | 北向/南向 | 当日资金余额 |

**处理逻辑:**
1. 过滤掉 `港股通` 行（保留 沪股通 + 深股通）
2. 判断 `资金方向`: 北向/入 = 流入, 南向/出 = 流出
3. 如果方向不明确，使用 `指数涨跌幅` 作为替代

---

## 3. 东方财富 (via AKShare) — 融资融券

### macro_china_market_margin_sh / sz

```python
df_sh = ak.macro_china_market_margin_sh()
df_sz = ak.macro_china_market_margin_sz()
```

**原始字段（中文）**:

| 字段 | 映射字段 | 说明 |
|------|----------|------|
| 日期 | `date` | 日期 |
| 融资余额 | `margin_balance` | 当日融资余额（元） |
| 融资买入额 | `buy_on_margin_value` | 当日融资买入（元） |
| 融券余额 | `short_balance` | 当日融券余额（元） |
| 融券卖出量 | `short_sell_quantity` | 当日融券卖出量 |

**数据量**: SH ~3941 行, SZ ~3743 行（2013年至今每个交易日）

### stock_margin_detail_sse / szse（备源）

```python
df_sse = ak.stock_margin_detail_sse(date="20260630")
df_szse = ak.stock_margin_detail_szse(date="20260630")
```

**原始字段（中文）** → 映射后:

| 原始字段 | 映射字段 |
|----------|----------|
| 股票代码 | `symbol` |
| 股票名称 | `name` |
| 融资余额 | `margin_balance` |
| 融资买入额 | `buy_on_margin_value` |
| 融资偿还额 | `margin_repayment` |
| 融券余量 | `short_balance_quantity` |
| 融券卖出量 | `short_sell_quantity` |
| 融券偿还量 | `short_repayment_quantity` |
| 融券余额 | `short_balance` |

**限制**: 仅返回单日数据，无历史。不带 `margin_type` 字段。

---

## 4. 容灾路径

```
融资融券数据:
  pdd.get_margin() ──┬── 成功 → df.attrs["source"] = "pandadata"
                     └── 失败 → ak.stock_margin_detail_sse/szse()
                                 → df.attrs["source"] = "eastmoney"

北向资金数据:
  ak.stock_hsgt_hist_em() ──┬── 成功, ak.stock_hsgt_fund_flow_summary_em() 成功
                             │   → df.attrs["source"] = "eastmoney"
                             ├── 仅汇总成功
                             │   → nb_summary.attrs["source"] = "degraded"
                             ├── 仅流向成功
                             │   → nb_flow.attrs["source"] = "degraded"
                             └── 全部失败
                                 → empty df, attrs["source"] = "degraded"

股票信息:
  pdd.get_trade_list() + pdd.get_stock_detail() ──┬── 成功
                                                    └── 失败 → 空 df（非阻塞）
```

## 5. 数据溯源标记

每个 DataFrame 携带来源元数据：

```python
df.attrs["source"]  # "pandadata" | "eastmoney" | "degraded"
```

报告的数据溯源表（第7部分）据此标注 ✅ / ⚠️ 状态。
