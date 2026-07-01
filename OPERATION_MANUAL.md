# 北向资金 + 融资融券全景监控 — 运维手册

## 目录

1. [环境配置](#环境配置)
2. [日常运行](#日常运行)
3. [信号解读](#信号解读)
4. [配置调优](#配置调优)
5. [故障排查](#故障排查)
6. [数据质量说明](#数据质量说明)

---

## 环境配置

### Pandadata 凭证

系统需要 Pandadata API 获取融资融券个股数据和股票信息。配置方式（二选一）：

**方式 1: 环境变量（推荐）**
```bash
export DEFAULT_USERNAME=8613800138000
export DEFAULT_PASSWORD=your_password
```

**方式 2: config.json**
```json
{
  "pandadata": {
    "username": "8613800138000",
    "password": "your_password",
    "base_url": "http://pandadata.pandaaiquant.com"
  }
}
```

> **注意**: 用户名必须加 `86` 前缀。如果 Pandadata 不可用，融资数据自动降级到东方财富。

### Python 环境

```bash
pip install -r requirements.txt
# panda_data >= 0.0.9
# akshare >= 1.18.0
# pandas >= 2.0.0
```

---

## 日常运行

### 基本命令

```bash
# 每个交易日收盘后运行（自动检测最新交易日）
python run.py

# 指定历史日期回溯
python run.py --date 20260630

# 强制重新获取数据（开发/调试用）
python run.py --date 20260630 --no-cache

# 查看详细日志
python run.py --verbose
```

### 定时任务

Windows Task Scheduler 或 Linux cron:

```bash
# 每个交易日 15:30 运行（cron 示例）
30 15 * * 1-5 cd /path/to/skill && python run.py >> logs/monitor.log 2>&1
```

### 缓存管理

```bash
# 查看已缓存的日期
# (缓存目录 cache/ 下的子目录名即为日期)

# 清理 30 天前的缓存
python run.py --cleanup-cache 30
```

缓存目录结构：
```
cache/
  20260630/
    northbound.parquet
    margin.parquet
    margin_detail.parquet
    stock_info.parquet
    .cache_meta.json
```

---

## 信号解读

### 北向资金信号

| 信号 | 触发条件 | 看多含义 | 看空含义 |
|------|----------|----------|----------|
| 净流向趋势 | 连续3日同向 | 外资持续流入 | 外资持续流出 |
| 单日异常 | Z-score ≥ 2.0 | 单日大幅抢筹 | 单日恐慌撤离 |
| 板块偏好 | 行业市值变动 | 重点行业获增持 | 重点行业被减持 |
| 重仓股变动 | 市值变动 > 0.5% | 重仓股加仓 | 重仓股减仓 |
| 累计趋势 | MA20/MA60 交叉 | 加速流入 | 加速流出 |
| 市场流向 | 沪深股通一致 | 两市一致流入 | 两市一致流出 |

> **重要**: 由于东方财富 API 近期 `net_buy_amount` 和 `market_value` 字段数据缺失，部分检测器自动降级使用 CSI300 指数变动作为替代指标。**市场流向**（沪股通/深股通方向）是目前最可靠的实时北向指标。

### 融资融券信号

| 信号 | 触发条件 | 看多含义 | 看空含义 |
|------|----------|----------|----------|
| 融资余额趋势 | 当前 vs MA20 | 温和上升（做多情绪） | 过热（>10%）或快速去杠杆 |
| 融资买入比 | vs 总成交 | 活跃（10-15%） | 过度杠杆（>15%） |
| 融券趋势 | 5日变化率 | 空头大幅回补 | 看空情绪上升 |
| 融资重仓股 | TOP20 集中度 | — | 集中度 > 40% 系统风险 |
| 融资融券比 | 比率趋势 | 多头主导 | 空头力量上升 |
| 融资类型分布 | 现金/股票占比 | — | 股票担保品过度集中 |

### 共振/背离模式

| 模式 | 北向 | 融资 | 含义 | 操作参考 |
|------|------|------|------|----------|
| 偏多共振 | 流入 ↑ | 扩张 ↑ | 内外资同步看多 | 顺势做多 |
| 偏空共振 | 流出 ↑ | 收缩 ↓ | 内外资同步看空 | 减仓/观望 |
| 谨慎偏多 | 流入 | 去杠杆 | 聪明钱逆势进场 | 关注底部信号 |
| 背离危险 | 流出 | 加杠杆 | 外资撤离散户接盘 | 高度警惕 |

### 综合评分解读

| 分数 | 等级 | 市场含义 |
|------|------|----------|
| 80-100 | A+, A, A- | 强烈看多 — 多处信号共振，资金面全面偏多 |
| 65-79 | B+, B, B- | 偏多 — 资金面总体向好 |
| 45-64 | C+, C, C- | 中性 — 多空均衡，观望为主 |
| 30-44 | D+, D, D- | 中性偏空 — 存在风险信号 |
| 0-29 | E+, E, E-, F | 看空/强烈看空 — 多重风险共振 |

---

## 配置调优

### 调整信号灵敏度

```json
{
  "northbound": {
    "consecutive_days_threshold": 3,   // 降低→更敏感，提高→更稳健
    "anomaly_zscore_threshold": 2.0,   // 标准正态 95% 置信 = 1.96
    "ma_short_window": 20,             // 短期均线周期
    "ma_long_window": 60               // 长期均线周期
  },
  "margin": {
    "buy_ratio_hot": 0.10,             // 融资活跃阈值
    "buy_ratio_dangerous": 0.15,       // 融资危险阈值
    "trend_short_window": 5,           // 短期趋势窗口
    "trend_long_window": 20            // 长期趋势窗口
  }
}
```

### 调整评分权重

```json
{
  "scoring": {
    "northbound_weight": 0.5,          // 北向权重（50%）
    "margin_weight": 0.5,              // 融资权重（50%）
    "risk_overheat_threshold": 0.7     // 风险惩罚触发阈值
  }
}
```

- 如果更关注外资动向：提高 `northbound_weight`
- 如果更关注杠杆情绪：提高 `margin_weight`

---

## 故障排查

### 常见问题

**1. "Pandadata credentials not configured"**

未配置 Pandadata 凭证。设置环境变量或 config.json 中的凭证。如果暂时无法获取 Pandadata 权限，系统会自动降级到东方财富数据源。

**2. "Empty stock universe for YYYYMMDD"**

Pandadata 对当天日期返回空数据（数据尚未结算）。使用前一个交易日：
```bash
python run.py --date 20260630  # 而非 20260701
```

**3. 融资买入比数值异常大**

已修复 — 旧版本将所有历史行求和导致数值膨胀。确认使用最新代码。

**4. 北向资金显示 "0亿" 或 "-"**

这是正常现象。东方财富 API 的 `market_value` 和 `net_buy_amount` 字段近期存在数据缺失。系统已自动降级使用 CSI300 指数变动作为替代。参考**市场流向**指标获取可靠的方向信号。

**5. 缓存保存失败**

```
WARNING Cache save failed (non-fatal): ...
```

不影响报告生成。原因可能是 Pandadata 返回的某些字段包含字符串 "NaN"。最新版本已加入自动清理逻辑。

**6. 网络超时**

数据采集涉及多轮 API 调用（5186 只股票分 26 批，4 线程并发），首次运行可能需要 3-5 分钟。后续使用缓存只需数秒。

### 日志级别

```bash
# 正常运行（INFO）
python run.py

# 调试模式（DEBUG）
python run.py --verbose
```

日志输出包含每个阶段的时间戳，方便定位慢查询。

---

## 数据质量说明

### 数据时延

| 数据 | 可用时间 | 备注 |
|------|----------|------|
| 融资融券个股 (Pandadata) | T日 20:00 后 | 需等交易所结算 |
| 融资融券宏观 (东方财富) | T日 18:00 后 | 通常较早可用 |
| 北向资金汇总 (东方财富) | T日 18:00 后 | 净买入字段可能为 NaN |
| 北向资金流向 (东方财富) | T日 18:00 后 | 沪股通/深股通方向可靠 |
| 股票信息 (Pandadata) | 实时 | 行业分类偶尔滞后 |

### 已知数据问题

1. **北向资金净买入 (net_buy_amount)**: 近约 2 周数据为 NaN，检测器使用 CSI300 替代
2. **北向持仓市值 (market_value)**: 2026年4月后全为 0
3. **个股北向持仓**: 不可用（东方财富 `stock_hsgt_individual_em` 数据停在 2024 年），板块分析基于汇总数据推算
4. **融资买入比**: 总成交额不可用，使用基于融资买入额的启发式估算

### 容灾路径

```
融资融券:
  Pandadata get_margin (主) ──fail──▶ 东方财富 SSE/SZSE (备)

北向资金:
  汇总 + 流向 (主) ──fail──▶ 仅汇总 (降级) ──fail──▶ 仅流向 (降级)

股票信息:
  Pandadata get_stock_detail (主) ──fail──▶ 空 DataFrame（非阻塞）
```

降级模式下报告会标注 `data.attrs["source"] = "degraded"`，数据溯源表会显示 ⚠️ 状态。
