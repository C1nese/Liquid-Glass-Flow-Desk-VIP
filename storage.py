from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from models import ExchangeSnapshot, LiquidationEvent, OrderBookQualityPoint, SpotSnapshot, TradeEvent


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=True, separators=(",", ":"))


def _empty_frame(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


class TerminalHistoryStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).with_name(".terminal_data")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.base_dir.joinpath("archive")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir.joinpath("terminal_history.sqlite3")
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    ts_ms INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    exchange_key TEXT NOT NULL,
                    exchange_name TEXT,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    last_price REAL,
                    mark_price REAL,
                    index_price REAL,
                    funding_bps REAL,
                    open_interest REAL,
                    open_interest_notional REAL,
                    volume_24h_notional REAL,
                    status TEXT,
                    error TEXT,
                    raw_json TEXT,
                    PRIMARY KEY (ts_ms, coin, exchange_key, market, symbol)
                );

                CREATE TABLE IF NOT EXISTS alert_events (
                    ts_ms INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    exchange_name TEXT NOT NULL,
                    exchange_key TEXT,
                    symbol TEXT,
                    level TEXT,
                    alert TEXT NOT NULL,
                    action TEXT NOT NULL,
                    explanation TEXT,
                    payload_json TEXT,
                    PRIMARY KEY (ts_ms, coin, exchange_name, alert, action)
                );

                CREATE TABLE IF NOT EXISTS market_events (
                    event_key TEXT PRIMARY KEY,
                    ts_ms INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    exchange_key TEXT NOT NULL,
                    exchange_name TEXT,
                    market TEXT NOT NULL,
                    category TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    price REAL,
                    size REAL,
                    notional REAL,
                    source TEXT,
                    raw_json TEXT
                );

                CREATE TABLE IF NOT EXISTS orderbook_quality_points (
                    point_key TEXT PRIMARY KEY,
                    ts_ms INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    exchange_key TEXT NOT NULL,
                    exchange_name TEXT,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    added_notional REAL,
                    canceled_notional REAL,
                    net_notional REAL,
                    near_added_notional REAL,
                    near_canceled_notional REAL,
                    spoof_events INTEGER,
                    refill_events INTEGER,
                    bid_wall_persistence_s REAL,
                    ask_wall_persistence_s REAL,
                    imbalance_pct REAL,
                    best_bid REAL,
                    best_ask REAL
                );

                CREATE INDEX IF NOT EXISTS idx_market_snapshots_coin_exchange_ts
                ON market_snapshots (coin, exchange_key, ts_ms);

                CREATE INDEX IF NOT EXISTS idx_alert_events_coin_ts
                ON alert_events (coin, ts_ms);

                CREATE INDEX IF NOT EXISTS idx_market_events_coin_category_ts
                ON market_events (coin, category, ts_ms);

                CREATE INDEX IF NOT EXISTS idx_orderbook_quality_coin_exchange_ts
                ON orderbook_quality_points (coin, exchange_key, ts_ms);
                """
            )

    def record_snapshots(self, coin: str, snapshots: Dict[str, ExchangeSnapshot | SpotSnapshot], *, market: str = "perp") -> int:
        rows: List[tuple] = []
        for exchange_key, snapshot in snapshots.items():
            timestamp_ms = int(snapshot.timestamp_ms or int(time.time() * 1000))
            rows.append(
                (
                    timestamp_ms,
                    str(coin or "").upper(),
                    str(exchange_key),
                    snapshot.exchange,
                    str(market),
                    snapshot.symbol,
                    snapshot.last_price,
                    getattr(snapshot, "mark_price", None),
                    getattr(snapshot, "index_price", None),
                    getattr(snapshot, "funding_bps", None),
                    getattr(snapshot, "open_interest", None),
                    getattr(snapshot, "open_interest_notional", None),
                    getattr(snapshot, "volume_24h_notional", None),
                    snapshot.status,
                    snapshot.error,
                    _json_dumps(getattr(snapshot, "raw", None)),
                )
            )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO market_snapshots (
                    ts_ms, coin, exchange_key, exchange_name, market, symbol,
                    last_price, mark_price, index_price, funding_bps, open_interest,
                    open_interest_notional, volume_24h_notional, status, error, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return int(connection.total_changes)

    def record_alert_timeline(
        self,
        coin: str,
        timeline_frame: pd.DataFrame,
        *,
        symbol_map: Dict[str, str] | None = None,
        exchange_title_map: Dict[str, str] | None = None,
    ) -> int:
        if timeline_frame.empty:
            return 0
        exchange_key_by_title = {str(title): str(key) for key, title in (exchange_title_map or {}).items()}
        rows: List[tuple] = []
        for row in timeline_frame.to_dict("records"):
            timestamp_value = row.get("时间")
            if timestamp_value is None or pd.isna(timestamp_value):
                continue
            timestamp_ms = int(pd.Timestamp(timestamp_value).timestamp() * 1000)
            exchange_name = str(row.get("交易所") or "未知")
            exchange_key = exchange_key_by_title.get(exchange_name)
            symbol = (symbol_map or {}).get(exchange_key or "", None)
            payload = {
                "level": row.get("等级"),
                "alert": row.get("告警"),
                "action": row.get("动作"),
                "explanation": row.get("说明"),
            }
            rows.append(
                (
                    timestamp_ms,
                    str(coin or "").upper(),
                    exchange_name,
                    exchange_key,
                    symbol,
                    row.get("等级"),
                    row.get("告警"),
                    row.get("动作"),
                    row.get("说明"),
                    _json_dumps(payload),
                )
            )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO alert_events (
                    ts_ms, coin, exchange_name, exchange_key, symbol,
                    level, alert, action, explanation, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return int(connection.total_changes)

    def record_market_events(
        self,
        coin: str,
        events_by_exchange: Dict[str, List[TradeEvent | LiquidationEvent]],
        *,
        category: str,
        exchange_title_map: Dict[str, str] | None = None,
        market: str = "perp",
    ) -> int:
        rows: List[tuple] = []
        for exchange_key, events in events_by_exchange.items():
            exchange_name = str((exchange_title_map or {}).get(exchange_key) or exchange_key)
            for event in events or []:
                event_key = "::".join(
                    [
                        str(category),
                        str(market),
                        str(exchange_key),
                        str(event.symbol),
                        str(event.timestamp_ms),
                        str(event.side or ""),
                        f"{float(event.price or 0.0):.8f}",
                        f"{float(event.size or 0.0):.8f}",
                    ]
                )
                rows.append(
                    (
                        event_key,
                        int(event.timestamp_ms),
                        str(coin or "").upper(),
                        str(exchange_key),
                        exchange_name,
                        str(market),
                        str(category),
                        str(event.symbol),
                        event.side,
                        event.price,
                        event.size,
                        event.notional,
                        getattr(event, "source", None),
                        _json_dumps(getattr(event, "raw", None)),
                    )
                )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO market_events (
                    event_key, ts_ms, coin, exchange_key, exchange_name, market,
                    category, symbol, side, price, size, notional, source, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return int(connection.total_changes)

    def record_quality_points(
        self,
        coin: str,
        *,
        exchange_key: str,
        exchange_name: str,
        symbol: str,
        points: List[OrderBookQualityPoint],
        market: str = "perp",
    ) -> int:
        rows: List[tuple] = []
        for point in points or []:
            point_key = "::".join(
                [
                    str(market),
                    str(exchange_key),
                    str(symbol),
                    str(point.timestamp_ms),
                    f"{float(point.added_notional or 0.0):.4f}",
                    f"{float(point.canceled_notional or 0.0):.4f}",
                ]
            )
            rows.append(
                (
                    point_key,
                    int(point.timestamp_ms),
                    str(coin or "").upper(),
                    str(exchange_key),
                    str(exchange_name),
                    str(market),
                    str(symbol),
                    point.added_notional,
                    point.canceled_notional,
                    point.net_notional,
                    point.near_added_notional,
                    point.near_canceled_notional,
                    int(point.spoof_events or 0),
                    int(point.refill_events or 0),
                    point.bid_wall_persistence_s,
                    point.ask_wall_persistence_s,
                    point.imbalance_pct,
                    point.best_bid,
                    point.best_ask,
                )
            )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO orderbook_quality_points (
                    point_key, ts_ms, coin, exchange_key, exchange_name, market, symbol,
                    added_notional, canceled_notional, net_notional, near_added_notional, near_canceled_notional,
                    spoof_events, refill_events, bid_wall_persistence_s, ask_wall_persistence_s,
                    imbalance_pct, best_bid, best_ask
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return int(connection.total_changes)

    def load_alert_events(self, *, coin: str | None = None, since_ms: int | None = None, limit: int = 500) -> pd.DataFrame:
        where: List[str] = []
        params: List[Any] = []
        if coin:
            where.append("coin = ?")
            params.append(str(coin).upper())
        if since_ms is not None:
            where.append("ts_ms >= ?")
            params.append(int(since_ms))
        sql = "SELECT ts_ms, coin, exchange_name, exchange_key, symbol, level, alert, action, explanation FROM alert_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_ms DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as connection:
            frame = pd.read_sql_query(sql, connection, params=params)
        if frame.empty:
            return _empty_frame(["时间", "币种", "交易所", "交易所键", "合约", "等级", "告警", "动作", "说明"])
        frame["时间"] = pd.to_datetime(frame["ts_ms"], unit="ms")
        return frame.rename(
            columns={
                "coin": "币种",
                "exchange_name": "交易所",
                "exchange_key": "交易所键",
                "symbol": "合约",
                "level": "等级",
                "alert": "告警",
                "action": "动作",
                "explanation": "说明",
            }
        )[["时间", "币种", "交易所", "交易所键", "合约", "等级", "告警", "动作", "说明"]]

    def load_market_history(
        self,
        *,
        coin: str | None = None,
        exchange_keys: Iterable[str] | None = None,
        market: str = "perp",
        since_ms: int | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        where = ["market = ?"]
        params: List[Any] = [market]
        if coin:
            where.append("coin = ?")
            params.append(str(coin).upper())
        selected_keys = [str(item) for item in (exchange_keys or []) if str(item)]
        if selected_keys:
            placeholders = ",".join("?" for _ in selected_keys)
            where.append(f"exchange_key IN ({placeholders})")
            params.extend(selected_keys)
        if since_ms is not None:
            where.append("ts_ms >= ?")
            params.append(int(since_ms))
        sql = (
            "SELECT ts_ms, coin, exchange_key, exchange_name, symbol, last_price, mark_price, index_price, "
            "funding_bps, open_interest, open_interest_notional, volume_24h_notional, status "
            "FROM market_snapshots WHERE "
            + " AND ".join(where)
            + " ORDER BY ts_ms DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._connect() as connection:
            frame = pd.read_sql_query(sql, connection, params=params)
        if frame.empty:
            return _empty_frame(["时间", "币种", "交易所键", "交易所", "合约", "最新价", "标记价", "指数价", "Funding(bps)", "OI", "OI金额", "24h成交额", "状态"])
        frame["时间"] = pd.to_datetime(frame["ts_ms"], unit="ms")
        return frame.rename(
            columns={
                "coin": "币种",
                "exchange_key": "交易所键",
                "exchange_name": "交易所",
                "symbol": "合约",
                "last_price": "最新价",
                "mark_price": "标记价",
                "index_price": "指数价",
                "funding_bps": "Funding(bps)",
                "open_interest": "OI",
                "open_interest_notional": "OI金额",
                "volume_24h_notional": "24h成交额",
                "status": "状态",
            }
        )[["时间", "币种", "交易所键", "交易所", "合约", "最新价", "标记价", "指数价", "Funding(bps)", "OI", "OI金额", "24h成交额", "状态"]]

    def load_market_events(
        self,
        *,
        coin: str | None = None,
        category: str | None = None,
        exchange_keys: Iterable[str] | None = None,
        market: str | None = None,
        since_ms: int | None = None,
        limit: int = 3000,
    ) -> pd.DataFrame:
        where: List[str] = []
        params: List[Any] = []
        if coin:
            where.append("coin = ?")
            params.append(str(coin).upper())
        if category:
            where.append("category = ?")
            params.append(str(category))
        if market:
            where.append("market = ?")
            params.append(str(market))
        selected_keys = [str(item) for item in (exchange_keys or []) if str(item)]
        if selected_keys:
            placeholders = ",".join("?" for _ in selected_keys)
            where.append(f"exchange_key IN ({placeholders})")
            params.extend(selected_keys)
        if since_ms is not None:
            where.append("ts_ms >= ?")
            params.append(int(since_ms))
        sql = "SELECT ts_ms, coin, exchange_key, exchange_name, market, category, symbol, side, price, size, notional, source FROM market_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_ms DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as connection:
            frame = pd.read_sql_query(sql, connection, params=params)
        if frame.empty:
            return _empty_frame(["时间", "币种", "交易所键", "交易所", "市场", "类型", "合约", "方向", "价格", "数量", "名义金额", "来源"])
        frame["时间"] = pd.to_datetime(frame["ts_ms"], unit="ms")
        return frame.rename(
            columns={
                "coin": "币种",
                "exchange_key": "交易所键",
                "exchange_name": "交易所",
                "market": "市场",
                "category": "类型",
                "symbol": "合约",
                "side": "方向",
                "price": "价格",
                "size": "数量",
                "notional": "名义金额",
                "source": "来源",
            }
        )[["时间", "币种", "交易所键", "交易所", "市场", "类型", "合约", "方向", "价格", "数量", "名义金额", "来源"]]

    def load_quality_history(
        self,
        *,
        coin: str | None = None,
        exchange_keys: Iterable[str] | None = None,
        market: str = "perp",
        since_ms: int | None = None,
        limit: int = 2000,
    ) -> pd.DataFrame:
        where = ["market = ?"]
        params: List[Any] = [market]
        if coin:
            where.append("coin = ?")
            params.append(str(coin).upper())
        selected_keys = [str(item) for item in (exchange_keys or []) if str(item)]
        if selected_keys:
            placeholders = ",".join("?" for _ in selected_keys)
            where.append(f"exchange_key IN ({placeholders})")
            params.extend(selected_keys)
        if since_ms is not None:
            where.append("ts_ms >= ?")
            params.append(int(since_ms))
        sql = (
            "SELECT ts_ms, coin, exchange_key, exchange_name, market, symbol, added_notional, canceled_notional, net_notional, "
            "near_added_notional, near_canceled_notional, spoof_events, refill_events, bid_wall_persistence_s, ask_wall_persistence_s, "
            "imbalance_pct, best_bid, best_ask FROM orderbook_quality_points WHERE "
            + " AND ".join(where)
            + " ORDER BY ts_ms DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._connect() as connection:
            frame = pd.read_sql_query(sql, connection, params=params)
        if frame.empty:
            return _empty_frame(["时间", "币种", "交易所键", "交易所", "市场", "合约", "新增挂单额", "撤单额", "净变化", "近价新增", "近价撤单", "假挂单次数", "补单次数", "买墙持续(s)", "卖墙持续(s)", "盘口失衡(%)", "最优买价", "最优卖价"])
        frame["时间"] = pd.to_datetime(frame["ts_ms"], unit="ms")
        return frame.rename(
            columns={
                "coin": "币种",
                "exchange_key": "交易所键",
                "exchange_name": "交易所",
                "market": "市场",
                "symbol": "合约",
                "added_notional": "新增挂单额",
                "canceled_notional": "撤单额",
                "net_notional": "净变化",
                "near_added_notional": "近价新增",
                "near_canceled_notional": "近价撤单",
                "spoof_events": "假挂单次数",
                "refill_events": "补单次数",
                "bid_wall_persistence_s": "买墙持续(s)",
                "ask_wall_persistence_s": "卖墙持续(s)",
                "imbalance_pct": "盘口失衡(%)",
                "best_bid": "最优买价",
                "best_ask": "最优卖价",
            }
        )[["时间", "币种", "交易所键", "交易所", "市场", "合约", "新增挂单额", "撤单额", "净变化", "近价新增", "近价撤单", "假挂单次数", "补单次数", "买墙持续(s)", "卖墙持续(s)", "盘口失衡(%)", "最优买价", "最优卖价"]]

    def describe(self) -> Dict[str, Any]:
        with self._connect() as connection:
            market_count = connection.execute("SELECT COUNT(1) FROM market_snapshots").fetchone()[0]
            alert_count = connection.execute("SELECT COUNT(1) FROM alert_events").fetchone()[0]
            event_count = connection.execute("SELECT COUNT(1) FROM market_events").fetchone()[0]
            quality_count = connection.execute("SELECT COUNT(1) FROM orderbook_quality_points").fetchone()[0]
            last_market_ts = connection.execute("SELECT MAX(ts_ms) FROM market_snapshots").fetchone()[0]
            last_alert_ts = connection.execute("SELECT MAX(ts_ms) FROM alert_events").fetchone()[0]
            last_event_ts = connection.execute("SELECT MAX(ts_ms) FROM market_events").fetchone()[0]
            last_quality_ts = connection.execute("SELECT MAX(ts_ms) FROM orderbook_quality_points").fetchone()[0]
        archive_files = sorted(self.archive_dir.rglob("*.*"))
        return {
            "db_path": str(self.db_path),
            "market_rows": int(market_count or 0),
            "alert_rows": int(alert_count or 0),
            "event_rows": int(event_count or 0),
            "quality_rows": int(quality_count or 0),
            "last_market_ts": int(last_market_ts) if last_market_ts else None,
            "last_alert_ts": int(last_alert_ts) if last_alert_ts else None,
            "last_event_ts": int(last_event_ts) if last_event_ts else None,
            "last_quality_ts": int(last_quality_ts) if last_quality_ts else None,
            "archive_files": [str(path) for path in archive_files[-16:]],
        }

    def archive_before(self, cutoff_ms: int, *, prefer_parquet: bool = True) -> List[Dict[str, Any]]:
        archived: List[Dict[str, Any]] = []
        for table_name in ("market_snapshots", "alert_events", "market_events", "orderbook_quality_points"):
            with self._connect() as connection:
                frame = pd.read_sql_query(
                    f"SELECT * FROM {table_name} WHERE ts_ms < ? ORDER BY ts_ms ASC",
                    connection,
                    params=[int(cutoff_ms)],
                )
            if frame.empty:
                continue
            frame["archive_day"] = pd.to_datetime(frame["ts_ms"], unit="ms").dt.strftime("%Y-%m-%d")
            for day_label, day_frame in frame.groupby("archive_day"):
                output_dir = self.archive_dir.joinpath(table_name)
                output_dir.mkdir(parents=True, exist_ok=True)
                clean_frame = day_frame.drop(columns=["archive_day"])
                output_path = output_dir.joinpath(f"{table_name}-{day_label}.parquet" if prefer_parquet else f"{table_name}-{day_label}.csv.gz")
                written_path = output_path
                if prefer_parquet:
                    try:
                        clean_frame.to_parquet(output_path, index=False)
                    except Exception:
                        written_path = output_dir.joinpath(f"{table_name}-{day_label}.csv.gz")
                        clean_frame.to_csv(written_path, index=False, compression="gzip")
                else:
                    clean_frame.to_csv(output_path, index=False, compression="gzip")
                archived.append({"table": table_name, "day": day_label, "rows": len(clean_frame), "path": str(written_path)})
            with self._connect() as connection:
                connection.execute(f"DELETE FROM {table_name} WHERE ts_ms < ?", (int(cutoff_ms),))
        return archived
