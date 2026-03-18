# Liquid Glass Flow Desk

多交易所现货 / 合约流动性终端，基于 Streamlit 构建，聚合 Binance、Bybit、OKX、Hyperliquid 的公开市场数据，并在同一界面里提供盘口、成交、OI、Funding、爆仓、地址模式、历史归档和告警复盘能力。

这个项目的目标不是做“单一图表页”，而是做一套更接近交易台的本地工作台：

- 同时看现货和合约
- 同时看单所和跨所
- 同时看静态快照、实时流和会话内历史
- 对未上架交易所自动跳过，避免无效请求和误判

## 当前能力

### 交易所覆盖

- `Binance`
  - 合约
  - 现货
  - 公开多空比 / taker 比 / OI / funding / liquidation
- `Bybit`
  - 合约
  - 现货
  - 公开 account ratio / OI / funding / liquidation
- `OKX`
  - 合约
  - 现货
  - OI / funding / market data / liquidation
- `Hyperliquid`
  - 合约
  - 地址模式
  - predicted fundings / all mids / vault / fills / funding / clearinghouseState

### 页面与功能

- `首页总览`
  - 快照总览
  - 现货 / 合约分账看板
  - OI / funding / basis / carry
  - 合约情绪真值层
  - 现货大额热力图
  - 合约大额热力图 + 清算热力图
  - 多币种异动榜

- `交易台深度页`
  - 合约深度
  - 现货深度
  - 综合深度
  - MBO 风格盘口画像
  - 风险热图
  - 盘口质量
  - 回放

- `告警中心`
  - 多因子告警
  - 置信度 / 样本数 / 数据新鲜度标签
  - 冷却时间与确认次数
  - 本地告警历史

- `爆仓中心`
  - 交易所实时爆仓流
  - 本地持久化归档
  - `最近 30 分钟 / 最近 4 小时 / 今天 / 全部本地缓存`
  - 簇识别和联动观察

- `盘口中心`
  - 假挂单 / 补单 / 撤单
  - 近价流动性塌陷
  - 盘口质量时间序列

- `增强实验室`
  - Hyperliquid 独有 API
  - 跨所聚合增强
  - 策略层实验
  - 跨币种联动
  - 通知与持久化

- `接口调试`
  - 当前快照
  - 实时状态
  - 原始 payload 检查

### Hyperliquid 地址模式

- 支持手动输入地址
- 支持公开示例地址快速测试
- 支持地址实时流
- 支持 `userFills / userFundings / userEvents / activeAssetData / clearinghouseState`
- 支持自定义地址池、地址分组和观察列表

### 历史与持久化

- SQLite 历史库
- 自动归档
  - 优先写 `parquet`
  - 环境不支持时自动回退到 `csv.gz`
- 本地爆仓归档
- 事件流 / 盘口质量 / 快照历史
- 告警复盘与命中率统计

### 通知

- 浏览器桌面通知
- 桌面通知音效
- Telegram 推送

## 币种选择逻辑

当前版本不是只支持几个常用币。

侧边栏的`全所币种搜索`会从各交易所公开 instruments 目录生成币种目录，并支持键盘搜索。你也可以继续通过`自定义币种`直接手动输入。

项目会对每个币种判断：

- 哪些交易所有`合约`
- 哪些交易所有`现货`

如果当前主图交易所没有这个币的合约市场：

- 会自动切到第一个可用交易所
- 页面会显示提示
- 未上架市场会被自动跳过，不再继续发无效请求

这意味着：

- 你可以查更多币
- 但不是每个币都一定四所全上
- UI 会尽量诚实地告诉你“哪里有，哪里没有”

## 运行方式

### 依赖

```bash
pip install -r requirements.txt
```

当前依赖：

- `streamlit`
- `requests`
- `pandas`
- `plotly`
- `websocket-client`

### 启动

```bash
streamlit run app.py
```

默认地址通常是：

- `http://localhost:8501`

## 建议环境

- Python `3.9+`
- 稳定网络
- 如果要长期运行历史归档，建议保留足够磁盘空间

## 使用建议

### 常规盯盘

建议优先使用：

- `当前交易所优先`
- `标准` 或 `轻量` 性能模式
- 顶部挂单统计先看 `当前交易所`

### 跨所测试

建议切到：

- `全部交易所`
- 顶部挂单统计 `四所聚合`
- 顶部挂单市场在 `合约 / 现货 / 合并` 间切换

### 非主流币 / 新币

建议：

- 先用`全所币种搜索`
- 如果命名不一致，再手动改`合约映射`
- 观察侧边栏里的市场覆盖提示

## 顶部挂单卡片说明

顶部卡片不是“全市场真实总挂单”。

它的口径是：

- 按当前设置的`盘口深度`
- 对每个交易所只统计请求到的前 `N` 档
- 可切 `当前交易所 / 四所聚合`
- 可切 `合约 / 现货 / 合并`

界面会显示：

- 买盘 / 卖盘挂单金额
- 买盘 / 卖盘挂单数量
- 实际采到的 `bid / ask` 档数

所以它更适合看：

- 当前可见簿面的压力
- 跨所对比
- 深度截断情况

不适合被理解成：

- 交易所完整全簿真值
- 隐藏单 / 冰山单 / 全市场未显示流动性

## 数据来源与口径

### 实时层

- WebSocket 优先
- REST 回补
- 会话内缓存

### 公开真值层

目前更接近“公开真值”的部分主要是：

- Binance / Bybit 的合约多空比
- OI
- funding
- liquidation

### 代理层

有些指标目前仍是代理口径，而不是官方全市场真值，例如：

- OKX / Hyperliquid 的合约情绪
- 部分清算密度推断
- 微结构异常与墙体行为识别

界面里已经尽量增加：

- `WS实时 / REST回补`
- 延迟
- 样本不足
- 置信度

## 本地文件

运行过程中会在本地产生这些数据：

- [`.terminal_ui_preferences.json`](E:/Codex/ex/.terminal_ui_preferences.json)
  - 记录 UI 偏好
  - 例如当前币种、主图交易所、范围模式、性能模式

- [`.terminal_data`](E:/Codex/ex/.terminal_data)
  - SQLite 历史库
  - 爆仓归档
  - 自动归档文件

主要文件包括：

- [`.terminal_data/terminal_history.sqlite3`](E:/Codex/ex/.terminal_data/terminal_history.sqlite3)
- [`.terminal_data/liquidations`](E:/Codex/ex/.terminal_data/liquidations)
- [`.terminal_data/archive`](E:/Codex/ex/.terminal_data/archive)

## 项目结构

- [app.py](E:/Codex/ex/app.py)
  - Streamlit 页面、缓存、编排和交互入口

- [analytics.py](E:/Codex/ex/analytics.py)
  - 指标计算
  - DataFrame 构造
  - Plotly 图表

- [exchanges.py](E:/Codex/ex/exchanges.py)
  - 各交易所 REST 适配
  - 符号映射
  - 目录抓取

- [realtime.py](E:/Codex/ex/realtime.py)
  - 实时服务
  - WebSocket
  - 会话内历史

- [storage.py](E:/Codex/ex/storage.py)
  - SQLite 历史库
  - 自动归档

- [models.py](E:/Codex/ex/models.py)
  - 数据模型

## 已知边界

- 本项目基于`公开 API`，不是私有账户终端
- 不包含下单能力
- 现货没有公开“多空比”真值，所以不要把现货看板理解成现货持仓情绪真值
- 不同交易所 liquidation 流的口径不完全一致
- 盘口热力图和风险热图属于`可解释推断图`，不是交易所内部清算引擎真值
- 不是每个币都在四所同时有现货和合约
- Hyperliquid 当前重点是合约和地址模式，现货公共整合仍弱于前三所

## 如果要继续扩展

这套项目现在最适合继续往这几条线推：

- 更强的告警路由
- 更深的历史库
- 更完整的地址池
- 更强的跨币种联动
- 更精细的执行质量 / 滑点模拟

## 快速排查

### 页面没数据

- 先看状态条
- 再看当前币种是否真的在该交易所上架
- 再看侧边栏`合约映射`

### 某个币只在部分交易所有

这是正常情况。

现在程序会：

- 自动跳过未上架市场
- 自动切换到可用主图交易所
- 保留可用市场的现货 / 合约数据

### Telegram 不工作

需要你自己填写：

- `Telegram Bot Token`
- `Telegram Chat ID`

这些凭据只保存在本地偏好文件里，不会上传到别处。
License
All rights reserved.

This project is proprietary and is provided for viewing and evaluation only. No part of this repository may be copied, modified, distributed, sublicensed, sold, or used for commercial purposes without prior written permission from the author.

If you need a commercial license, please contact me.

授权说明
本项目及其全部代码、设计和内容版权归作者所有，保留所有权利。

未经作者事先书面许可，任何人不得对本项目进行：

商业使用
复制分发
修改后发布
二次售卖
集成到付费产品或服务中
如需商业授权，请联系作者。
