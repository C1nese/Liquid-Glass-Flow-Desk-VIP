from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional

import websocket

from exchanges import (
    EXCHANGE_ORDER,
    SPOT_EXCHANGE_ORDER,
    build_clients,
    build_spot_clients,
    compute_notional,
    fetch_binance_futures_orderbook_snapshot,
    fetch_binance_spot_orderbook_snapshot,
    fetch_hyperliquid_address_mode,
    is_valid_onchain_address,
    normalize_liquidation_side,
    normalize_onchain_address,
    normalize_trade_side,
    safe_float,
    safe_int,
)
from models import (
    ExchangeSnapshot,
    LiquidationEvent,
    OIPoint,
    OrderBookLevel,
    OrderBookQualityPoint,
    RecordedMarketEvent,
    SpotSnapshot,
    TradeEvent,
)


class LocalLiquidationArchive:
    def __init__(self, base_dir: Optional[Path] = None, retention_hours: int = 24 * 14) -> None:
        self.base_dir = base_dir or Path(__file__).with_name(".terminal_data").joinpath("liquidations")
        self.retention_hours = max(retention_hours, 24)
        self._last_prune_ms: Dict[str, int] = {}
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(symbol or "").upper()) or "UNKNOWN"

    @staticmethod
    def _event_key(event: LiquidationEvent) -> tuple:
        return (
            event.exchange,
            event.symbol,
            event.timestamp_ms,
            event.side,
            round(event.price or 0.0, 6),
            round(event.size or 0.0, 6),
        )

    def _symbol_dir(self, exchange_key: str, symbol: str) -> Path:
        return self.base_dir / exchange_key / self._safe_symbol(symbol)

    def _event_path(self, exchange_key: str, event: LiquidationEvent) -> Path:
        day_key = time.strftime("%Y-%m-%d", time.gmtime(max(int(event.timestamp_ms), 0) / 1000.0))
        return self._symbol_dir(exchange_key, event.symbol) / f"{day_key}.jsonl"

    def _maybe_prune(self, symbol_dir: Path, now_ms: int) -> None:
        cache_key = str(symbol_dir)
        last_prune_ms = self._last_prune_ms.get(cache_key, 0)
        if now_ms - last_prune_ms < 30 * 60_000:
            return
        self._last_prune_ms[cache_key] = now_ms
        cutoff_ms = now_ms - self.retention_hours * 3_600_000
        try:
            for path in symbol_dir.glob("*.jsonl"):
                try:
                    if int(path.stat().st_mtime * 1000) < cutoff_ms:
                        path.unlink(missing_ok=True)
                except OSError:
                    continue
        except OSError:
            return

    def append(self, exchange_key: str, event: LiquidationEvent) -> None:
        path = self._event_path(exchange_key, event)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "exchange": event.exchange,
                "symbol": event.symbol,
                "timestamp_ms": int(event.timestamp_ms),
                "side": event.side,
                "price": event.price,
                "size": event.size,
                "notional": event.notional,
                "source": event.source,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except OSError:
            return
        self._maybe_prune(path.parent, int(time.time() * 1000))

    def load(
        self,
        exchange_key: str,
        symbol: str,
        *,
        since_ms: Optional[int] = None,
        limit: int = 4000,
    ) -> List[LiquidationEvent]:
        symbol_dir = self._symbol_dir(exchange_key, symbol)
        if not symbol_dir.exists():
            return []
        events: List[LiquidationEvent] = []
        seen = set()
        paths = sorted(symbol_dir.glob("*.jsonl"))
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                timestamp_ms = safe_int(payload.get("timestamp_ms")) or 0
                if since_ms is not None and timestamp_ms < since_ms:
                    continue
                event = LiquidationEvent(
                    exchange=str(payload.get("exchange") or exchange_key.title()),
                    symbol=str(payload.get("symbol") or symbol),
                    timestamp_ms=timestamp_ms,
                    side=normalize_liquidation_side(payload.get("side")),
                    price=safe_float(payload.get("price")),
                    size=safe_float(payload.get("size")),
                    notional=safe_float(payload.get("notional")),
                    source=str(payload.get("source") or "persisted"),
                )
                event_key = self._event_key(event)
                if event_key in seen:
                    continue
                seen.add(event_key)
                events.append(event)
        events.sort(key=lambda item: item.timestamp_ms)
        if limit > 0 and len(events) > limit:
            return events[-limit:]
        return events

    def describe(
        self,
        exchange_key: str,
        symbol: str,
        *,
        since_ms: Optional[int] = None,
        limit: int = 4000,
    ) -> Dict[str, Any]:
        events = self.load(exchange_key, symbol, since_ms=since_ms, limit=limit)
        symbol_dir = self._symbol_dir(exchange_key, symbol)
        return {
            "count": len(events),
            "first_timestamp_ms": events[0].timestamp_ms if events else None,
            "last_timestamp_ms": events[-1].timestamp_ms if events else None,
            "path": str(symbol_dir),
        }


class HyperliquidAddressStreamService:
    def __init__(
        self,
        address: str,
        coin: str,
        timeout: int = 10,
        lookback_hours: int = 24,
        fill_history_size: int = 500,
        funding_history_size: int = 240,
        event_history_size: int = 500,
    ) -> None:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        self.address = normalized_address
        self.coin = str(coin or "").strip().upper()
        self.timeout = timeout
        self.lookback_hours = max(1, int(lookback_hours))
        self.fill_history_size = max(100, fill_history_size)
        self.funding_history_size = max(60, funding_history_size)
        self.event_history_size = max(120, event_history_size)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ws_app: websocket.WebSocketApp | None = None
        self.thread: threading.Thread | None = None
        self.fills: Deque[Dict[str, Any]] = deque(maxlen=self.fill_history_size)
        self.fundings: Deque[Dict[str, Any]] = deque(maxlen=self.funding_history_size)
        self.user_events: Deque[Dict[str, Any]] = deque(maxlen=self.event_history_size)
        self.positions: List[Dict[str, Any]] = []
        self.active_asset: Dict[str, Any] = {}
        self.raw_state: Dict[str, Any] = {}
        self.role: str | None = None
        self.portfolio: List[dict] = []
        self.vault_equities: List[dict] = []
        self.vault_details: Dict[str, Any] = {}
        self.account_value: float | None = None
        self.total_margin_used: float | None = None
        self.withdrawable: float | None = None
        self.total_notional_position: float | None = None
        self.connected = False
        self.stream_status = "bootstrapping"
        self.error: str | None = None
        self.last_message_ms: int | None = None
        self.last_snapshot_ms: int | None = None
        self._hydrate_from_rest()
        self._start_thread()

    @staticmethod
    def _fill_key(item: Dict[str, Any]) -> tuple:
        return (
            str(item.get("coin") or ""),
            int(item.get("time") or 0),
            str(item.get("hash") or ""),
            round(float(item.get("price") or 0.0), 6),
            round(float(item.get("size") or 0.0), 6),
        )

    @staticmethod
    def _funding_key(item: Dict[str, Any]) -> tuple:
        return (
            str(item.get("coin") or ""),
            int(item.get("time") or 0),
            round(float(item.get("amount") or 0.0), 6),
            str(item.get("type") or ""),
        )

    @staticmethod
    def _parse_fill(item: Dict[str, Any]) -> Dict[str, Any]:
        price = safe_float(item.get("px")) or safe_float(item.get("price"))
        size = safe_float(item.get("sz")) or safe_float(item.get("size"))
        return {
            "time": safe_int(item.get("time")) or 0,
            "coin": str(item.get("coin") or ""),
            "direction": str(item.get("dir") or item.get("direction") or item.get("side") or ""),
            "side": normalize_trade_side(item.get("side")),
            "price": price,
            "size": size,
            "notional": compute_notional(price, size),
            "closed_pnl": safe_float(item.get("closedPnl")) or safe_float(item.get("closed_pnl")),
            "fee": safe_float(item.get("fee")),
            "fee_token": str(item.get("feeToken") or item.get("fee_token") or ""),
            "start_position": safe_float(item.get("startPosition")) or safe_float(item.get("start_position")),
            "hash": str(item.get("hash") or ""),
            "liquidation": item.get("liquidation"),
            "raw": item,
        }

    @staticmethod
    def _parse_funding(item: Dict[str, Any]) -> Dict[str, Any]:
        amount = safe_float(item.get("usdc")) or safe_float(item.get("amount"))
        return {
            "time": safe_int(item.get("time")) or 0,
            "coin": str(item.get("coin") or ""),
            "amount": amount,
            "type": str(item.get("type") or "funding"),
            "direction": "received" if amount is not None and amount >= 0 else "paid",
            "funding_rate": safe_float(item.get("fundingRate")) or safe_float(item.get("funding_rate")),
            "size": safe_float(item.get("szi")) or safe_float(item.get("size")),
            "raw": item,
        }

    @staticmethod
    def _parse_position(item: Dict[str, Any]) -> Dict[str, Any]:
        position = item.get("position") if isinstance(item.get("position"), dict) else item
        signed_size = safe_float(position.get("szi")) or safe_float(position.get("sz")) or safe_float(position.get("signedSz"))
        entry_price = safe_float(position.get("entryPx"))
        mark_price = safe_float(position.get("markPx"))
        role_side = "flat"
        if signed_size is not None:
            if signed_size > 0:
                role_side = "long"
            elif signed_size < 0:
                role_side = "short"
        return {
            "coin": str(position.get("coin") or item.get("coin") or ""),
            "side": role_side,
            "size": abs(signed_size) if signed_size is not None else None,
            "signed_size": signed_size,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "liquidation_price": safe_float(position.get("liquidationPx")),
            "leverage": safe_float((position.get("leverage") or {}).get("value") if isinstance(position.get("leverage"), dict) else position.get("leverage")),
            "position_value": safe_float(position.get("positionValue")),
            "unrealized_pnl": safe_float(position.get("unrealizedPnl")),
            "return_on_equity": safe_float(position.get("returnOnEquity")),
            "raw": position,
        }

    def _hydrate_from_rest(self) -> None:
        try:
            bundle = fetch_hyperliquid_address_mode(self.address, self.coin, self.lookback_hours, timeout=self.timeout)
        except Exception as exc:
            with self.lock:
                self.stream_status = "error"
                self.error = str(exc)
            return
        with self.lock:
            self.account_value = safe_float(bundle.get("account_value"))
            self.total_margin_used = safe_float(bundle.get("total_margin_used"))
            self.withdrawable = safe_float(bundle.get("withdrawable"))
            self.total_notional_position = safe_float(bundle.get("total_notional_position"))
            self.active_asset = dict(bundle.get("active_asset") or {})
            self.raw_state = dict(bundle.get("raw_state") or {})
            self.role = str(bundle.get("role") or "") or None
            self.portfolio = list(bundle.get("portfolio") or [])
            self.vault_equities = list(bundle.get("vault_equities") or [])
            self.vault_details = dict(bundle.get("vault_details") or {})
            self.last_snapshot_ms = safe_int(bundle.get("timestamp_ms")) or int(time.time() * 1000)
            self.error = str(bundle.get("error") or "") or None
            self._replace_positions_locked(list(bundle.get("positions") or []))
            self._replace_fills_locked(list(bundle.get("fills") or []))
            self._replace_fundings_locked(list(bundle.get("funding") or []))

    def _start_thread(self) -> None:
        self.thread = threading.Thread(target=self._run_worker, name=f"ws-hyper-user-{self.address[:10]}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.ws_app is not None:
            try:
                self.ws_app.close()
            except Exception:
                pass

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": "ok" if self.stream_status != "error" else "error",
                "address": self.address,
                "coin": self.coin,
                "account_value": self.account_value,
                "total_margin_used": self.total_margin_used,
                "withdrawable": self.withdrawable,
                "total_notional_position": self.total_notional_position,
                "positions": list(self.positions),
                "fills": list(self.fills),
                "funding": list(self.fundings),
                "user_events": list(self.user_events),
                "active_asset": dict(self.active_asset),
                "raw_state": dict(self.raw_state),
                "role": self.role,
                "portfolio": list(self.portfolio),
                "vault_equities": list(self.vault_equities),
                "vault_details": dict(self.vault_details),
                "stream_status": self.stream_status,
                "connected": self.connected,
                "last_message_ms": self.last_message_ms,
                "timestamp_ms": self.last_snapshot_ms or self.last_message_ms,
                "error": self.error,
            }

    def get_transport_health(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "connected": self.connected,
                "stream_status": self.stream_status,
                "last_message_ms": self.last_message_ms,
                "last_snapshot_ms": self.last_snapshot_ms,
                "fill_count": len(self.fills),
                "funding_count": len(self.fundings),
                "event_count": len(self.user_events),
                "error": self.error,
            }

    def _replace_positions_locked(self, positions: List[Dict[str, Any]]) -> None:
        selected = positions
        if self.coin:
            selected = [item for item in positions if str(item.get("coin") or "").upper() == self.coin]
        selected = sorted(selected, key=lambda item: abs(float(item.get("position_value") or 0.0)), reverse=True)
        self.positions = selected

    def _append_fill_locked(self, fill: Dict[str, Any]) -> None:
        if self.coin and str(fill.get("coin") or "").upper() != self.coin:
            return
        key = self._fill_key(fill)
        if any(self._fill_key(existing) == key for existing in self.fills):
            return
        self.fills.append(fill)

    def _replace_fills_locked(self, fills: List[Dict[str, Any]]) -> None:
        self.fills.clear()
        for item in sorted(fills, key=lambda value: int(value.get("time") or 0)):
            self._append_fill_locked(item)

    def _append_funding_locked(self, funding: Dict[str, Any]) -> None:
        if self.coin and str(funding.get("coin") or "").upper() != self.coin:
            return
        key = self._funding_key(funding)
        if any(self._funding_key(existing) == key for existing in self.fundings):
            return
        self.fundings.append(funding)

    def _replace_fundings_locked(self, funding_items: List[Dict[str, Any]]) -> None:
        self.fundings.clear()
        for item in sorted(funding_items, key=lambda value: int(value.get("time") or 0)):
            self._append_funding_locked(item)

    def _append_user_event_locked(self, category: str, payload: Dict[str, Any]) -> None:
        event = {
            "time": int(payload.get("time") or payload.get("timestamp_ms") or int(time.time() * 1000)),
            "category": category,
            "payload": payload,
        }
        event_key = (event["time"], category, json.dumps(payload, sort_keys=True, ensure_ascii=True))
        if any(
            (existing.get("time"), existing.get("category"), json.dumps(existing.get("payload"), sort_keys=True, ensure_ascii=True)) == event_key
            for existing in self.user_events
        ):
            return
        self.user_events.append(event)

    def _run_worker(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                self.stream_status = "connecting"
            app = websocket.WebSocketApp(
                "wss://api.hyperliquid.xyz/ws",
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.ws_app = app
            try:
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._on_error(app, exc)
            if self.stop_event.wait(3):
                return

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        subscriptions = [
            {"type": "clearinghouseState", "user": self.address},
            {"type": "userFills", "user": self.address, "aggregateByTime": True},
            {"type": "userFundings", "user": self.address},
            {"type": "userEvents", "user": self.address},
            {"type": "webData3", "user": self.address},
        ]
        if self.coin:
            subscriptions.append({"type": "activeAssetData", "user": self.address, "coin": self.coin})
        for subscription in subscriptions:
            ws.send(json.dumps({"method": "subscribe", "subscription": subscription}))
        with self.lock:
            self.connected = True
            self.stream_status = "live"
            self.error = None

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        channel = str(payload.get("channel") or "")
        data = payload.get("data")
        with self.lock:
            self.connected = True
            self.stream_status = "live"
            self.error = None
            self.last_message_ms = int(time.time() * 1000)
        if channel == "subscriptionResponse":
            return
        if channel == "userFills":
            self._handle_user_fills(data)
            return
        if channel == "userFundings":
            self._handle_user_fundings(data)
            return
        if channel == "userEvents":
            self._handle_user_events(data)
            return
        if channel == "clearinghouseState":
            self._handle_state(data)
            return
        if channel == "activeAssetData":
            self._handle_active_asset_data(data)
            return
        if channel == "webData3":
            self._handle_web_data(data)

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        with self.lock:
            self.connected = False
            self.stream_status = "degraded"
            self.error = str(error)

    def _on_close(self, ws: websocket.WebSocketApp, close_status_code: object, close_message: object) -> None:
        with self.lock:
            self.connected = False
            if not self.stop_event.is_set() and self.stream_status != "error":
                self.stream_status = "reconnecting"

    def _handle_user_fills(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        fills = data.get("fills") or []
        if isinstance(fills, dict):
            fills = [fills]
        parsed = [self._parse_fill(item) for item in fills if isinstance(item, dict)]
        with self.lock:
            if bool(data.get("isSnapshot")):
                self._replace_fills_locked(parsed)
            else:
                for item in parsed:
                    self._append_fill_locked(item)

    def _handle_user_fundings(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        funding_items = data.get("fundings") or data.get("fundingPayments") or []
        if not funding_items and all(key in data for key in ("time", "coin")):
            funding_items = [data]
        if isinstance(funding_items, dict):
            funding_items = [funding_items]
        parsed = [self._parse_funding(item) for item in funding_items if isinstance(item, dict)]
        with self.lock:
            if bool(data.get("isSnapshot")):
                self._replace_fundings_locked(parsed)
            else:
                for item in parsed:
                    self._append_funding_locked(item)

    def _handle_user_events(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        with self.lock:
            if isinstance(data.get("fills"), list):
                for item in data.get("fills") or []:
                    if isinstance(item, dict):
                        parsed_fill = self._parse_fill(item)
                        self._append_fill_locked(parsed_fill)
                        self._append_user_event_locked("fills", parsed_fill)
            if isinstance(data.get("funding"), dict):
                parsed_funding = self._parse_funding(data.get("funding") or {})
                self._append_funding_locked(parsed_funding)
                self._append_user_event_locked("funding", parsed_funding)
            if isinstance(data.get("liquidation"), dict):
                payload = dict(data.get("liquidation") or {})
                payload["time"] = payload.get("time") or int(time.time() * 1000)
                self._append_user_event_locked("liquidation", payload)
            if isinstance(data.get("nonUserCancel"), list):
                for item in data.get("nonUserCancel") or []:
                    if isinstance(item, dict):
                        payload = dict(item)
                        payload["time"] = payload.get("time") or int(time.time() * 1000)
                        self._append_user_event_locked("nonUserCancel", payload)

    def _handle_state(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        positions = [self._parse_position(item) for item in data.get("assetPositions") or [] if isinstance(item, dict)]
        margin_summary = data.get("marginSummary") or {}
        with self.lock:
            self.account_value = safe_float(margin_summary.get("accountValue"))
            self.total_margin_used = safe_float(margin_summary.get("totalMarginUsed"))
            self.total_notional_position = safe_float(margin_summary.get("totalNtlPos"))
            self.withdrawable = safe_float(data.get("withdrawable"))
            self.raw_state = dict(data)
            self.last_snapshot_ms = safe_int(data.get("time")) or int(time.time() * 1000)
            self._replace_positions_locked(positions)

    def _handle_active_asset_data(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        with self.lock:
            self.active_asset = dict(data)

    def _handle_web_data(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        user_state = data.get("userState") if isinstance(data.get("userState"), dict) else {}
        perp_states = data.get("perpDexStates") if isinstance(data.get("perpDexStates"), list) else []
        leading_vaults: List[dict] = []
        total_vault_equity = 0.0
        for state in perp_states:
            if not isinstance(state, dict):
                continue
            for item in state.get("leadingVaults") or []:
                if isinstance(item, dict):
                    leading_vaults.append(item)
            total_vault_equity += float(safe_float(state.get("totalVaultEquity")) or 0.0)
        with self.lock:
            if not self.role:
                self.role = "vault" if bool(user_state.get("isVault")) else self.role
            if leading_vaults:
                self.vault_details.setdefault("leadingVaults", leading_vaults)
            if total_vault_equity > 0:
                self.vault_details.setdefault("totalVaultEquity", total_vault_equity)


class LiveTerminalService:
    def __init__(
        self,
        symbol_map: Dict[str, str],
        timeout: int = 10,
        sample_seconds: int = 15,
        history_size: int = 720,
        liquidation_history_size: int = 600,
        trade_history_size: int = 1200,
        record_history_size: int = 2400,
        orderbook_limit: int = 80,
        spot_symbol: str = "BTCUSDT",
        spot_symbol_map: Dict[str, str] | None = None,
    ) -> None:
        self.symbol_map = dict(symbol_map)
        self.spot_symbol_map = dict(spot_symbol_map or {"binance": spot_symbol})
        self.timeout = timeout
        self.sample_seconds = max(sample_seconds, 5)
        self.history_size = max(history_size, 120)
        self.liquidation_history_size = max(liquidation_history_size, 120)
        self.trade_history_size = max(trade_history_size, 240)
        self.record_history_size = max(record_history_size, 480)
        self.orderbook_limit = max(orderbook_limit, 20)
        self.clients = build_clients(timeout=timeout)
        self.spot_clients = build_spot_clients(timeout=timeout)
        self.liquidation_archive = LocalLiquidationArchive()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.live_by_exchange: Dict[str, ExchangeSnapshot] = {}
        self.sampled_by_exchange: Dict[str, ExchangeSnapshot] = {}
        self.oi_history: Dict[str, Deque[OIPoint]] = {
            key: deque(maxlen=self.history_size) for key in EXCHANGE_ORDER
        }
        self.liquidation_history: Dict[str, Deque[LiquidationEvent]] = {
            key: deque(maxlen=self.liquidation_history_size) for key in EXCHANGE_ORDER
        }
        self.trade_history: Dict[str, Deque[TradeEvent]] = {
            key: deque(maxlen=self.trade_history_size) for key in EXCHANGE_ORDER
        }
        self.orderbook_quality_history: Dict[str, Deque[OrderBookQualityPoint]] = {
            key: deque(maxlen=self.history_size) for key in EXCHANGE_ORDER
        }
        self.recorded_events: Dict[str, Deque[RecordedMarketEvent]] = {
            key: deque(maxlen=self.record_history_size) for key in EXCHANGE_ORDER
        }
        self.orderbooks: Dict[str, List[OrderBookLevel]] = {key: [] for key in EXCHANGE_ORDER}
        self.orderbook_state: Dict[str, Dict[str, Dict[float, float]]] = {
            key: {"bid": {}, "ask": {}} for key in EXCHANGE_ORDER
        }
        self.orderbook_quality_state: Dict[str, Dict[str, object]] = {
            key: self._initial_book_quality_state() for key in EXCHANGE_ORDER
        }
        self.binance_depth_buffer: Deque[Dict[str, object]] = deque(maxlen=1200)
        self.binance_depth_last_u: int | None = None
        self.binance_depth_synced = False
        self.binance_depth_bootstrapping = False
        self.ws_apps: Dict[str, websocket.WebSocketApp] = {}
        self.threads: List[threading.Thread] = []
        self.spot_snapshots: Dict[str, SpotSnapshot | None] = {
            key: None for key in SPOT_EXCHANGE_ORDER
        }
        self.spot_orderbooks: Dict[str, List[OrderBookLevel]] = {
            key: [] for key in SPOT_EXCHANGE_ORDER
        }
        self.spot_trade_histories: Dict[str, Deque[TradeEvent]] = {
            key: deque(maxlen=self.trade_history_size) for key in SPOT_EXCHANGE_ORDER
        }
        self.spot_orderbook_quality_history: Dict[str, Deque[OrderBookQualityPoint]] = {
            key: deque(maxlen=self.history_size) for key in SPOT_EXCHANGE_ORDER
        }
        self.spot_recorded_events: Dict[str, Deque[RecordedMarketEvent]] = {
            key: deque(maxlen=self.record_history_size) for key in SPOT_EXCHANGE_ORDER
        }
        self.spot_orderbook_state: Dict[str, Dict[str, Dict[float, float]]] = {
            key: {"bid": {}, "ask": {}} for key in SPOT_EXCHANGE_ORDER
        }
        self.spot_orderbook_quality_state: Dict[str, Dict[str, object]] = {
            key: self._initial_book_quality_state() for key in SPOT_EXCHANGE_ORDER
        }
        self.binance_spot_depth_buffer: Deque[Dict[str, object]] = deque(maxlen=1200)
        self.binance_spot_depth_last_u: int | None = None
        self.binance_spot_depth_synced = False
        self.binance_spot_depth_bootstrapping = False

        self._sample_once()
        self._start_threads()

    def _start_threads(self) -> None:
        sampler = threading.Thread(target=self._run_sampler, name="oi-sampler", daemon=True)
        sampler.start()
        self.threads.append(sampler)

        for exchange_key in EXCHANGE_ORDER:
            if not self.symbol_map.get(exchange_key):
                continue
            worker = threading.Thread(
                target=self._run_ws_worker,
                args=(exchange_key,),
                name=f"ws-{exchange_key}",
                daemon=True,
            )
            worker.start()
            self.threads.append(worker)

        for exchange_key in SPOT_EXCHANGE_ORDER:
            if not self.spot_symbol_map.get(exchange_key):
                continue
            spot_worker = threading.Thread(
                target=self._run_spot_ws_worker,
                args=(exchange_key,),
                name=f"ws-spot-{exchange_key}",
                daemon=True,
            )
            spot_worker.start()
            self.threads.append(spot_worker)

    def stop(self) -> None:
        self.stop_event.set()
        for ws in list(self.ws_apps.values()):
            try:
                ws.close()
            except Exception:
                pass

    def current_snapshots(self) -> List[ExchangeSnapshot]:
        snapshots: List[ExchangeSnapshot] = []
        with self.lock:
            for exchange_key in EXCHANGE_ORDER:
                live = self.live_by_exchange.get(exchange_key)
                sampled = self.sampled_by_exchange.get(exchange_key)
                exchange_name = self.clients[exchange_key].exchange_name
                symbol = self.symbol_map.get(exchange_key, "")

                if not symbol:
                    snapshots.append(
                        ExchangeSnapshot(
                            exchange=exchange_name,
                            symbol="",
                            status="error",
                            error="未上架此币",
                        )
                    )
                    continue

                if sampled is not None:
                    merged = replace(sampled)
                elif live is not None:
                    merged = replace(live)
                else:
                    merged = ExchangeSnapshot(
                        exchange=exchange_name,
                        symbol=symbol,
                        status="error",
                        error="waiting for data",
                    )

                if live is not None:
                    for field_name in (
                        "last_price",
                        "mark_price",
                        "index_price",
                        "open_interest",
                        "open_interest_notional",
                        "funding_rate",
                        "volume_24h_base",
                        "volume_24h_notional",
                        "timestamp_ms",
                    ):
                        live_value = getattr(live, field_name)
                        if live_value is not None:
                            setattr(merged, field_name, live_value)
                    if live.status == "ok":
                        merged.status = "ok"
                        merged.error = None

                snapshots.append(merged)
        return snapshots

    def get_oi_history(self, exchange_key: str) -> List[OIPoint]:
        with self.lock:
            return list(self.oi_history.get(exchange_key, []))

    def get_liquidation_history(self, exchange_key: str) -> List[LiquidationEvent]:
        with self.lock:
            return list(self.liquidation_history.get(exchange_key, []))

    def get_persisted_liquidations(
        self,
        exchange_key: str,
        symbol: str,
        *,
        since_ms: Optional[int] = None,
        limit: int = 4000,
    ) -> List[LiquidationEvent]:
        with self.lock:
            return self.liquidation_archive.load(exchange_key, symbol, since_ms=since_ms, limit=limit)

    def get_persisted_liquidation_summary(
        self,
        exchange_key: str,
        symbol: str,
        *,
        since_ms: Optional[int] = None,
        limit: int = 4000,
    ) -> Dict[str, Any]:
        with self.lock:
            return self.liquidation_archive.describe(exchange_key, symbol, since_ms=since_ms, limit=limit)

    def get_trade_history(self, exchange_key: str) -> List[TradeEvent]:
        with self.lock:
            return list(self.trade_history.get(exchange_key, []))

    def get_orderbook_quality_history(self, exchange_key: str) -> List[OrderBookQualityPoint]:
        with self.lock:
            return list(self.orderbook_quality_history.get(exchange_key, []))

    def get_recorded_events(self, exchange_key: str) -> List[RecordedMarketEvent]:
        with self.lock:
            return list(self.recorded_events.get(exchange_key, []))

    def get_orderbook(self, exchange_key: str) -> List[OrderBookLevel]:
        with self.lock:
            return list(self.orderbooks.get(exchange_key, []))

    def ensure_orderbook_limit(self, limit: int) -> None:
        target_limit = max(int(limit), 20)
        with self.lock:
            if target_limit == self.orderbook_limit:
                return
            self.orderbook_limit = target_limit
            for exchange_key in EXCHANGE_ORDER:
                self._refresh_orderbook_levels_locked(exchange_key)
            for exchange_key in SPOT_EXCHANGE_ORDER:
                self._refresh_spot_orderbook_locked(exchange_key)

    def get_spot_snapshot(self, exchange_key: str = "binance") -> SpotSnapshot | None:
        with self.lock:
            snapshot = self.spot_snapshots.get(exchange_key)
            return replace(snapshot) if snapshot is not None else None

    def get_spot_orderbook(self, exchange_key: str = "binance") -> List[OrderBookLevel]:
        with self.lock:
            return list(self.spot_orderbooks.get(exchange_key, []))

    def get_spot_trade_history(self, exchange_key: str = "binance") -> List[TradeEvent]:
        with self.lock:
            return list(self.spot_trade_histories.get(exchange_key, []))

    def get_spot_orderbook_quality_history(self, exchange_key: str = "binance") -> List[OrderBookQualityPoint]:
        with self.lock:
            return list(self.spot_orderbook_quality_history.get(exchange_key, []))

    def get_spot_recorded_events(self, exchange_key: str = "binance") -> List[RecordedMarketEvent]:
        with self.lock:
            return list(self.spot_recorded_events.get(exchange_key, []))

    def get_transport_health(self, exchange_key: str, *, spot: bool = False) -> Dict[str, object]:
        with self.lock:
            if spot:
                if not self.spot_symbol_map.get(exchange_key):
                    return {
                        "snapshot_timestamp_ms": None,
                        "trade_timestamp_ms": None,
                        "orderbook_levels": 0,
                        "sync_state": "unsupported",
                    }
                snapshot = self.spot_snapshots.get(exchange_key)
                orderbook = self.spot_orderbooks.get(exchange_key, [])
                trades = self.spot_trade_histories.get(exchange_key, [])
                if exchange_key == "binance":
                    sync_state = (
                        "synced"
                        if self.binance_spot_depth_synced
                        else "bootstrapping"
                        if self.binance_spot_depth_bootstrapping
                        else "degraded"
                    )
                else:
                    sync_state = "synced" if orderbook else "waiting"
                return {
                    "snapshot_timestamp_ms": snapshot.timestamp_ms if snapshot is not None else None,
                    "trade_timestamp_ms": max((event.timestamp_ms for event in trades), default=None),
                    "orderbook_levels": len(orderbook),
                    "sync_state": sync_state,
                }

            live = self.live_by_exchange.get(exchange_key)
            sampled = self.sampled_by_exchange.get(exchange_key)
            orderbook = self.orderbooks.get(exchange_key, [])
            trades = self.trade_history.get(exchange_key, [])
            if not self.symbol_map.get(exchange_key):
                return {
                    "snapshot_timestamp_ms": None,
                    "live_timestamp_ms": None,
                    "sample_timestamp_ms": None,
                    "trade_timestamp_ms": None,
                    "orderbook_levels": 0,
                    "sync_state": "unsupported",
                }
            if exchange_key == "binance":
                sync_state = (
                    "synced"
                    if self.binance_depth_synced
                    else "bootstrapping"
                    if self.binance_depth_bootstrapping
                    else "degraded"
                )
            else:
                sync_state = "synced" if orderbook else "waiting"
            snapshot_timestamp_ms = None
            if live is not None and live.timestamp_ms is not None:
                snapshot_timestamp_ms = live.timestamp_ms
            elif sampled is not None:
                snapshot_timestamp_ms = sampled.timestamp_ms
            return {
                "snapshot_timestamp_ms": snapshot_timestamp_ms,
                "live_timestamp_ms": live.timestamp_ms if live is not None else None,
                "sample_timestamp_ms": sampled.timestamp_ms if sampled is not None else None,
                "trade_timestamp_ms": max((event.timestamp_ms for event in trades), default=None),
                "orderbook_levels": len(orderbook),
                "sync_state": sync_state,
            }

    @staticmethod
    def _initial_book_quality_state() -> Dict[str, object]:
        return {
            "wall_registry": {"bid": {}, "ask": {}},
            "refill_watch": {"bid": None, "ask": None},
        }

    def _run_sampler(self) -> None:
        while not self.stop_event.is_set():
            self._sample_once()
            if self.stop_event.wait(self.sample_seconds):
                return

    def _sample_once(self) -> None:
        for exchange_key in SPOT_EXCHANGE_ORDER:
            symbol = self.spot_symbol_map.get(exchange_key)
            if not symbol:
                continue
            try:
                spot_snapshot = self.spot_clients[exchange_key].fetch(symbol)
            except Exception as exc:
                spot_snapshot = SpotSnapshot(
                    exchange=self.spot_clients[exchange_key].exchange_name,
                    symbol=symbol,
                    status="error",
                    error=str(exc),
                )
            with self.lock:
                self.spot_snapshots[exchange_key] = spot_snapshot
                if self.spot_orderbooks.get(exchange_key):
                    self._sync_spot_best_prices_locked(exchange_key)

        for exchange_key in EXCHANGE_ORDER:
            symbol = self.symbol_map.get(exchange_key)
            if not symbol:
                continue
            try:
                snapshot = self.clients[exchange_key].fetch(symbol)
            except Exception as exc:
                snapshot = ExchangeSnapshot(
                    exchange=self.clients[exchange_key].exchange_name,
                    symbol=symbol,
                    status="error",
                    error=str(exc),
                )

            with self.lock:
                self.sampled_by_exchange[exchange_key] = snapshot
                if snapshot.status == "ok":
                    self._append_oi_point_locked(exchange_key, snapshot)
                    self.live_by_exchange.setdefault(exchange_key, replace(snapshot))

    def _append_oi_point_locked(self, exchange_key: str, snapshot: ExchangeSnapshot) -> None:
        if snapshot.open_interest is None and snapshot.open_interest_notional is None:
            return
        history = self.oi_history[exchange_key]
        point = OIPoint(
            timestamp_ms=snapshot.timestamp_ms or int(time.time() * 1000),
            open_interest=snapshot.open_interest,
            open_interest_notional=snapshot.open_interest_notional,
        )
        if history and abs(history[-1].timestamp_ms - point.timestamp_ms) <= 1_000:
            history[-1] = point
        else:
            history.append(point)
        self._append_recorded_event_locked(
            exchange_key,
            RecordedMarketEvent(
                timestamp_ms=point.timestamp_ms,
                exchange=snapshot.exchange,
                symbol=snapshot.symbol,
                category="oi",
                market="perp",
                value=point.open_interest_notional if point.open_interest_notional is not None else point.open_interest,
                label="OI update",
                raw={
                    "open_interest": point.open_interest,
                    "open_interest_notional": point.open_interest_notional,
                },
            ),
        )

    def _append_liquidation_event_locked(self, exchange_key: str, event: LiquidationEvent) -> None:
        history = self.liquidation_history[exchange_key]
        event_id = (
            event.timestamp_ms,
            event.side,
            round(event.price or 0.0, 6),
            round(event.size or 0.0, 6),
        )
        if history:
            last = history[-1]
            last_id = (
                last.timestamp_ms,
                last.side,
                round(last.price or 0.0, 6),
                round(last.size or 0.0, 6),
            )
            if last_id == event_id:
                return
        history.append(event)
        self.liquidation_archive.append(exchange_key, event)
        self._append_recorded_event_locked(
            exchange_key,
            RecordedMarketEvent(
                timestamp_ms=event.timestamp_ms,
                exchange=event.exchange,
                symbol=event.symbol,
                category="liquidation",
                market="perp",
                side=event.side,
                price=event.price,
                size=event.size,
                notional=event.notional,
                label=event.source,
                raw=event.raw,
            ),
        )

    def _append_trade_event_locked(self, exchange_key: str, event: TradeEvent) -> None:
        history = self.trade_history[exchange_key]
        event_id = (
            event.timestamp_ms,
            event.side,
            round(event.price or 0.0, 6),
            round(event.size or 0.0, 6),
        )
        if history:
            last = history[-1]
            last_id = (
                last.timestamp_ms,
                last.side,
                round(last.price or 0.0, 6),
                round(last.size or 0.0, 6),
            )
            if last_id == event_id:
                return
        history.append(event)
        self._append_recorded_event_locked(
            exchange_key,
            RecordedMarketEvent(
                timestamp_ms=event.timestamp_ms,
                exchange=event.exchange,
                symbol=event.symbol,
                category="trade",
                market="perp",
                side=event.side,
                price=event.price,
                size=event.size,
                notional=event.notional,
                label=event.source,
                raw=event.raw,
            ),
        )

    def _append_recorded_event_locked(self, exchange_key: str, event: RecordedMarketEvent, *, spot: bool = False) -> None:
        history = self.spot_recorded_events[exchange_key] if spot else self.recorded_events[exchange_key]
        event_id = (
            event.timestamp_ms,
            event.category,
            event.market,
            event.side,
            round(event.price or 0.0, 6),
            round(event.value or event.notional or 0.0, 6),
        )
        if history:
            last = history[-1]
            last_id = (
                last.timestamp_ms,
                last.category,
                last.market,
                last.side,
                round(last.price or 0.0, 6),
                round(last.value or last.notional or 0.0, 6),
            )
            if last_id == event_id:
                return
        history.append(event)

    @staticmethod
    def _compute_book_imbalance(bid_state: Dict[float, float], ask_state: Dict[float, float]) -> float | None:
        bid_notional = sum(price * size for price, size in bid_state.items() if size > 0)
        ask_notional = sum(price * size for price, size in ask_state.items() if size > 0)
        total_notional = bid_notional + ask_notional
        if total_notional <= 0:
            return None
        return (bid_notional - ask_notional) / total_notional * 100.0

    @staticmethod
    def _best_prices_from_state(side_state: Dict[str, Dict[float, float]]) -> tuple[float | None, float | None]:
        bid_prices = [price for price, size in side_state["bid"].items() if size > 0]
        ask_prices = [price for price, size in side_state["ask"].items() if size > 0]
        best_bid = max(bid_prices) if bid_prices else None
        best_ask = min(ask_prices) if ask_prices else None
        return best_bid, best_ask

    @staticmethod
    def _reference_price_from_state(side_state: Dict[str, Dict[float, float]]) -> float | None:
        best_bid, best_ask = LiveTerminalService._best_prices_from_state(side_state)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) * 0.5
        return best_bid if best_bid is not None else best_ask

    def _record_orderbook_quality_locked(
        self,
        exchange_key: str,
        previous_bid_state: Dict[float, float],
        previous_ask_state: Dict[float, float],
        *,
        spot: bool = False,
    ) -> None:
        side_state = self.spot_orderbook_state[exchange_key] if spot else self.orderbook_state[exchange_key]
        quality_history = self.spot_orderbook_quality_history[exchange_key] if spot else self.orderbook_quality_history[exchange_key]
        quality_state = self.spot_orderbook_quality_state[exchange_key] if spot else self.orderbook_quality_state[exchange_key]
        exchange_name = self.spot_clients[exchange_key].exchange_name if spot else self.clients[exchange_key].exchange_name
        symbol = self.spot_symbol_map.get(exchange_key, "") if spot else self.symbol_map.get(exchange_key, "")
        now_ms = int(time.time() * 1000)
        reference_price = self._reference_price_from_state(side_state)
        best_bid, best_ask = self._best_prices_from_state(side_state)
        added_notional = 0.0
        canceled_notional = 0.0
        near_added_notional = 0.0
        near_canceled_notional = 0.0
        spoof_events = 0
        refill_events = 0
        persistence_by_side = {"bid": 0.0, "ask": 0.0}
        near_ratio = 0.0016
        spoof_window_ms = 8_000
        refill_window_ms = 3_000

        for side in ("bid", "ask"):
            previous_state = previous_bid_state if side == "bid" else previous_ask_state
            current_state = side_state[side]
            prices = set(previous_state) | set(current_state)
            for price in prices:
                previous_size = previous_state.get(price, 0.0)
                current_size = current_state.get(price, 0.0)
                delta_size = current_size - previous_size
                if abs(delta_size) <= 1e-12:
                    continue
                delta_notional = abs(delta_size) * price
                is_near = reference_price is not None and reference_price > 0 and abs(price - reference_price) / reference_price <= near_ratio
                if delta_size > 0:
                    added_notional += delta_notional
                    if is_near:
                        near_added_notional += delta_notional
                else:
                    canceled_notional += delta_notional
                    if is_near:
                        near_canceled_notional += delta_notional

            wall_registry = quality_state["wall_registry"][side]
            current_levels = [(price, size, price * size) for price, size in current_state.items() if size > 0]
            notionals = sorted((notional for _, _, notional in current_levels), reverse=True)
            threshold = notionals[min(2, len(notionals) - 1)] if notionals else 0.0
            if threshold <= 0 and notionals:
                threshold = notionals[0]
            notable_prices = set()
            for price, size, notional in current_levels:
                if notional < threshold or threshold <= 0:
                    continue
                notable_prices.add(price)
                if price not in wall_registry:
                    wall_registry[price] = {
                        "first_seen": now_ms,
                        "last_seen": now_ms,
                        "peak_notional": notional,
                    }
                else:
                    wall_registry[price]["last_seen"] = now_ms
                    wall_registry[price]["peak_notional"] = max(float(wall_registry[price]["peak_notional"]), notional)

            expired_prices = []
            for price, metadata in wall_registry.items():
                if price in notable_prices:
                    continue
                age_ms = now_ms - int(metadata["first_seen"])
                if age_ms <= spoof_window_ms:
                    spoof_events += 1
                expired_prices.append(price)
            for price in expired_prices:
                wall_registry.pop(price, None)

            live_ages = [max(0.0, (now_ms - int(metadata["first_seen"])) / 1000.0) for metadata in wall_registry.values()]
            persistence_by_side[side] = sum(live_ages) / len(live_ages) if live_ages else 0.0

            previous_best_price = max(previous_state) if side == "bid" and previous_state else min(previous_state) if previous_state else None
            current_best_price = best_bid if side == "bid" else best_ask
            previous_best_size = previous_state.get(previous_best_price, 0.0) if previous_best_price is not None else 0.0
            current_best_size = current_state.get(current_best_price, 0.0) if current_best_price is not None else 0.0
            refill_watch = quality_state["refill_watch"][side]
            if previous_best_price is not None and previous_best_size > 0:
                same_zone = (
                    current_best_price is not None
                    and abs(current_best_price - previous_best_price) / previous_best_price <= 0.0008
                )
                if same_zone and current_best_size < previous_best_size * 0.65:
                    quality_state["refill_watch"][side] = {
                        "timestamp_ms": now_ms,
                        "price": current_best_price,
                        "target_size": previous_best_size,
                    }
                    refill_watch = quality_state["refill_watch"][side]
            if refill_watch is not None:
                if now_ms - int(refill_watch["timestamp_ms"]) > refill_window_ms:
                    quality_state["refill_watch"][side] = None
                elif (
                    current_best_price is not None
                    and abs(current_best_price - float(refill_watch["price"])) / max(float(refill_watch["price"]), 1e-9) <= 0.0008
                    and current_best_size >= float(refill_watch["target_size"]) * 0.85
                ):
                    refill_events += 1
                    quality_state["refill_watch"][side] = None

        quality_point = OrderBookQualityPoint(
            timestamp_ms=now_ms,
            added_notional=added_notional,
            canceled_notional=canceled_notional,
            net_notional=added_notional - canceled_notional,
            near_added_notional=near_added_notional,
            near_canceled_notional=near_canceled_notional,
            spoof_events=spoof_events,
            refill_events=refill_events,
            bid_wall_persistence_s=persistence_by_side["bid"],
            ask_wall_persistence_s=persistence_by_side["ask"],
            imbalance_pct=self._compute_book_imbalance(side_state["bid"], side_state["ask"]),
            best_bid=best_bid,
            best_ask=best_ask,
        )
        if quality_history and abs(quality_history[-1].timestamp_ms - quality_point.timestamp_ms) <= 800:
            previous = quality_history[-1]
            quality_history[-1] = OrderBookQualityPoint(
                timestamp_ms=quality_point.timestamp_ms,
                added_notional=previous.added_notional + quality_point.added_notional,
                canceled_notional=previous.canceled_notional + quality_point.canceled_notional,
                net_notional=previous.net_notional + quality_point.net_notional,
                near_added_notional=previous.near_added_notional + quality_point.near_added_notional,
                near_canceled_notional=previous.near_canceled_notional + quality_point.near_canceled_notional,
                spoof_events=previous.spoof_events + quality_point.spoof_events,
                refill_events=previous.refill_events + quality_point.refill_events,
                bid_wall_persistence_s=quality_point.bid_wall_persistence_s,
                ask_wall_persistence_s=quality_point.ask_wall_persistence_s,
                imbalance_pct=quality_point.imbalance_pct,
                best_bid=quality_point.best_bid,
                best_ask=quality_point.best_ask,
            )
        else:
            quality_history.append(quality_point)

        self._append_recorded_event_locked(
            exchange_key,
            RecordedMarketEvent(
                timestamp_ms=quality_point.timestamp_ms,
                exchange=exchange_name,
                symbol=symbol,
                category="orderbook_quality",
                market="spot" if spot else "perp",
                value=quality_point.net_notional,
                label="quality",
                raw={
                    "added_notional": quality_point.added_notional,
                    "canceled_notional": quality_point.canceled_notional,
                    "spoof_events": quality_point.spoof_events,
                    "refill_events": quality_point.refill_events,
                    "imbalance_pct": quality_point.imbalance_pct,
                },
            ),
            spot=spot,
        )

    def _replace_orderbook_locked(self, exchange_key: str, bids, asks) -> None:
        bid_state = self.orderbook_state[exchange_key]["bid"]
        ask_state = self.orderbook_state[exchange_key]["ask"]
        previous_bid_state = dict(bid_state)
        previous_ask_state = dict(ask_state)
        bid_state.clear()
        ask_state.clear()
        for price, size in bids:
            if price is not None and size is not None and size > 0:
                bid_state[price] = size
        for price, size in asks:
            if price is not None and size is not None and size > 0:
                ask_state[price] = size
        self._refresh_orderbook_levels_locked(exchange_key)
        self._record_orderbook_quality_locked(exchange_key, previous_bid_state, previous_ask_state)

    def _apply_orderbook_delta_locked(self, exchange_key: str, bids, asks) -> None:
        bid_state = self.orderbook_state[exchange_key]["bid"]
        ask_state = self.orderbook_state[exchange_key]["ask"]
        previous_bid_state = dict(bid_state)
        previous_ask_state = dict(ask_state)
        for price, size in bids:
            if price is None or size is None:
                continue
            if size <= 0:
                bid_state.pop(price, None)
            else:
                bid_state[price] = size
        for price, size in asks:
            if price is None or size is None:
                continue
            if size <= 0:
                ask_state.pop(price, None)
            else:
                ask_state[price] = size
        self._refresh_orderbook_levels_locked(exchange_key)
        self._record_orderbook_quality_locked(exchange_key, previous_bid_state, previous_ask_state)

    def _refresh_orderbook_levels_locked(self, exchange_key: str) -> None:
        bid_levels = sorted(
            self.orderbook_state[exchange_key]["bid"].items(),
            key=lambda item: item[0],
            reverse=True,
        )[: self.orderbook_limit]
        ask_levels = sorted(
            self.orderbook_state[exchange_key]["ask"].items(),
            key=lambda item: item[0],
        )[: self.orderbook_limit]
        self.orderbooks[exchange_key] = [
            OrderBookLevel(price=price, size=size, side="bid") for price, size in bid_levels
        ] + [
            OrderBookLevel(price=price, size=size, side="ask") for price, size in ask_levels
        ]

    def _replace_spot_orderbook_locked(self, exchange_key: str, bids, asks) -> None:
        previous_bid_state = dict(self.spot_orderbook_state[exchange_key]["bid"])
        previous_ask_state = dict(self.spot_orderbook_state[exchange_key]["ask"])
        self.spot_orderbook_state[exchange_key]["bid"].clear()
        self.spot_orderbook_state[exchange_key]["ask"].clear()
        for price, size in bids:
            if price is not None and size is not None and size > 0:
                self.spot_orderbook_state[exchange_key]["bid"][price] = size
        for price, size in asks:
            if price is not None and size is not None and size > 0:
                self.spot_orderbook_state[exchange_key]["ask"][price] = size
        self._refresh_spot_orderbook_locked(exchange_key)
        self._record_orderbook_quality_locked(exchange_key, previous_bid_state, previous_ask_state, spot=True)

    def _apply_spot_orderbook_delta_locked(self, exchange_key: str, bids, asks) -> None:
        previous_bid_state = dict(self.spot_orderbook_state[exchange_key]["bid"])
        previous_ask_state = dict(self.spot_orderbook_state[exchange_key]["ask"])
        for price, size in bids:
            if price is None or size is None:
                continue
            if size <= 0:
                self.spot_orderbook_state[exchange_key]["bid"].pop(price, None)
            else:
                self.spot_orderbook_state[exchange_key]["bid"][price] = size
        for price, size in asks:
            if price is None or size is None:
                continue
            if size <= 0:
                self.spot_orderbook_state[exchange_key]["ask"].pop(price, None)
            else:
                self.spot_orderbook_state[exchange_key]["ask"][price] = size
        self._refresh_spot_orderbook_locked(exchange_key)
        self._record_orderbook_quality_locked(exchange_key, previous_bid_state, previous_ask_state, spot=True)

    def _refresh_spot_orderbook_locked(self, exchange_key: str) -> None:
        bid_levels = sorted(
            self.spot_orderbook_state[exchange_key]["bid"].items(),
            key=lambda item: item[0],
            reverse=True,
        )[: self.orderbook_limit]
        ask_levels = sorted(
            self.spot_orderbook_state[exchange_key]["ask"].items(),
            key=lambda item: item[0],
        )[: self.orderbook_limit]
        self.spot_orderbooks[exchange_key] = [OrderBookLevel(price=price, size=size, side="bid") for price, size in bid_levels] + [
            OrderBookLevel(price=price, size=size, side="ask") for price, size in ask_levels
        ]
        self._sync_spot_best_prices_locked(exchange_key)

    def _sync_spot_best_prices_locked(self, exchange_key: str) -> None:
        top_bid = next((level.price for level in self.spot_orderbooks[exchange_key] if level.side == "bid"), None)
        top_ask = next((level.price for level in self.spot_orderbooks[exchange_key] if level.side == "ask"), None)
        if self.spot_snapshots.get(exchange_key) is None:
            self.spot_snapshots[exchange_key] = SpotSnapshot(
                exchange=self.spot_clients[exchange_key].exchange_name,
                symbol=self.spot_symbol_map.get(exchange_key, ""),
            )
        self.spot_snapshots[exchange_key].bid_price = top_bid
        self.spot_snapshots[exchange_key].ask_price = top_ask

    def _append_spot_trade_event_locked(self, exchange_key: str, event: TradeEvent) -> None:
        event_id = (
            event.timestamp_ms,
            event.side,
            round(event.price or 0.0, 6),
            round(event.size or 0.0, 6),
        )
        history = self.spot_trade_histories[exchange_key]
        if history:
            last = history[-1]
            last_id = (
                last.timestamp_ms,
                last.side,
                round(last.price or 0.0, 6),
                round(last.size or 0.0, 6),
            )
            if last_id == event_id:
                return
        history.append(event)
        self._append_recorded_event_locked(
            exchange_key,
            RecordedMarketEvent(
                timestamp_ms=event.timestamp_ms,
                exchange=event.exchange,
                symbol=event.symbol,
                category="trade",
                market="spot",
                side=event.side,
                price=event.price,
                size=event.size,
                notional=event.notional,
                label=event.source,
                raw=event.raw,
            ),
            spot=True,
        )

    def _run_ws_worker(self, exchange_key: str) -> None:
        while not self.stop_event.is_set():
            symbol = self.symbol_map.get(exchange_key)
            if not symbol:
                return
            ws_url = self._build_ws_url(exchange_key, symbol)
            app = websocket.WebSocketApp(
                ws_url,
                on_open=lambda ws, key=exchange_key, sym=symbol: self._on_open(key, sym, ws),
                on_message=lambda ws, message, key=exchange_key, sym=symbol: self._on_message(key, sym, message),
                on_error=lambda ws, error, key=exchange_key, sym=symbol: self._on_error(key, sym, error),
            )
            self.ws_apps[exchange_key] = app
            try:
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._on_error(exchange_key, symbol, exc)
            if self.stop_event.wait(3):
                return

    def _run_spot_ws_worker(self, exchange_key: str) -> None:
        while not self.stop_event.is_set():
            symbol = self.spot_symbol_map.get(exchange_key)
            if not symbol:
                return
            ws_url = self._build_spot_ws_url(exchange_key, symbol)
            app = websocket.WebSocketApp(
                ws_url,
                on_open=lambda ws, key=exchange_key, sym=symbol: self._on_spot_open(key, sym, ws),
                on_message=lambda ws, message, key=exchange_key, sym=symbol: self._on_spot_message(key, sym, message),
                on_error=lambda ws, error, key=exchange_key, sym=symbol: self._on_spot_error(key, sym, error),
            )
            self.ws_apps[f"spot_{exchange_key}"] = app
            try:
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._on_spot_error(exchange_key, symbol, exc)
            if self.stop_event.wait(3):
                return

    def _build_ws_url(self, exchange_key: str, symbol: str) -> str:
        if exchange_key == "bybit":
            return "wss://stream.bybit.com/v5/public/linear"
        if exchange_key == "binance":
            lower = symbol.lower()
            return (
                "wss://fstream.binance.com/stream?streams="
                f"{lower}@markPrice@1s/{lower}@ticker/{lower}@forceOrder/{lower}@aggTrade/{lower}@depth@100ms"
            )
        if exchange_key == "okx":
            return "wss://ws.okx.com:8443/ws/v5/public"
        return "wss://api.hyperliquid.xyz/ws"

    def _build_spot_ws_url(self, exchange_key: str, symbol: str) -> str:
        if exchange_key == "binance":
            lower = symbol.lower()
            return f"wss://stream.binance.com:9443/stream?streams={lower}@ticker/{lower}@aggTrade/{lower}@depth@100ms"
        if exchange_key == "bybit":
            return "wss://stream.bybit.com/v5/public/spot"
        return "wss://ws.okx.com:8443/ws/v5/public"

    def _on_open(self, exchange_key: str, symbol: str, ws: websocket.WebSocketApp) -> None:
        if exchange_key == "bybit":
            ws.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "args": [
                            f"tickers.{symbol}",
                            f"allLiquidation.{symbol}",
                            f"orderbook.50.{symbol}",
                            f"publicTrade.{symbol}",
                        ],
                    }
                )
            )
        elif exchange_key == "okx":
            ws.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "args": [
                            {"channel": "tickers", "instId": symbol},
                            {"channel": "mark-price", "instId": symbol},
                            {"channel": "books5", "instId": symbol},
                            {"channel": "trades", "instId": symbol},
                        ],
                    }
                )
            )
        elif exchange_key == "hyperliquid":
            ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}}))
            ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": symbol}}))
            ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": symbol}}))

    def _on_spot_open(self, exchange_key: str, symbol: str, ws: websocket.WebSocketApp) -> None:
        if exchange_key == "bybit":
            ws.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "args": [
                            f"tickers.{symbol}",
                            f"orderbook.50.{symbol}",
                            f"publicTrade.{symbol}",
                        ],
                    }
                )
            )
        elif exchange_key == "okx":
            ws.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "args": [
                            {"channel": "tickers", "instId": symbol},
                            {"channel": "books5", "instId": symbol},
                            {"channel": "trades", "instId": symbol},
                        ],
                    }
                )
            )

    def _on_message(self, exchange_key: str, symbol: str, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        if exchange_key == "bybit":
            topic = str(payload.get("topic", ""))
            if topic.startswith("allLiquidation"):
                self._handle_bybit_liquidation(symbol, payload)
            elif topic.startswith("orderbook"):
                self._handle_bybit_orderbook(payload)
            elif topic.startswith("publicTrade"):
                self._handle_bybit_trade(symbol, payload)
            else:
                self._handle_bybit_message(symbol, payload)
        elif exchange_key == "binance":
            self._handle_binance_message(symbol, payload)
        elif exchange_key == "okx":
            self._handle_okx_message(symbol, payload)
        elif exchange_key == "hyperliquid":
            self._handle_hyperliquid_message(symbol, payload)

    def _on_spot_message(self, exchange_key: str, symbol: str, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        self._handle_spot_message(exchange_key, symbol, payload)

    def _on_error(self, exchange_key: str, symbol: str, error: Exception) -> None:
        with self.lock:
            previous = self.live_by_exchange.get(exchange_key)
            snapshot = replace(previous) if previous is not None else ExchangeSnapshot(
                exchange=self.clients[exchange_key].exchange_name,
                symbol=symbol,
            )
            snapshot.status = "error"
            snapshot.error = str(error)
            self.live_by_exchange[exchange_key] = snapshot

    def _on_spot_error(self, exchange_key: str, symbol: str, error: Exception) -> None:
        with self.lock:
            previous = self.spot_snapshots.get(exchange_key)
            snapshot = replace(previous) if previous is not None else SpotSnapshot(
                exchange=self.spot_clients[exchange_key].exchange_name,
                symbol=symbol,
            )
            snapshot.status = "error"
            snapshot.error = str(error)
            self.spot_snapshots[exchange_key] = snapshot

    def _update_live_snapshot(self, exchange_key: str, symbol: str, **values) -> None:
        with self.lock:
            previous = self.live_by_exchange.get(exchange_key)
            snapshot = replace(previous) if previous is not None else ExchangeSnapshot(
                exchange=self.clients[exchange_key].exchange_name,
                symbol=symbol,
            )
            snapshot.exchange = self.clients[exchange_key].exchange_name
            snapshot.symbol = symbol
            snapshot.status = "ok"
            snapshot.error = None
            for key, value in values.items():
                if value is not None:
                    setattr(snapshot, key, value)
            self.live_by_exchange[exchange_key] = snapshot

    def _update_spot_snapshot(self, exchange_key: str, symbol: str, **values) -> None:
        with self.lock:
            previous = self.spot_snapshots.get(exchange_key)
            snapshot = replace(previous) if previous is not None else SpotSnapshot(
                exchange=self.spot_clients[exchange_key].exchange_name,
                symbol=symbol,
            )
            snapshot.exchange = self.spot_clients[exchange_key].exchange_name
            snapshot.symbol = symbol
            snapshot.status = "ok"
            snapshot.error = None
            for key, value in values.items():
                if value is not None:
                    setattr(snapshot, key, value)
            self.spot_snapshots[exchange_key] = snapshot

    def _handle_bybit_message(self, symbol: str, payload: Dict[str, object]) -> None:
        if payload.get("success") is not None:
            return
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return
        self._update_live_snapshot(
            "bybit",
            symbol,
            last_price=safe_float(data.get("lastPrice")),
            mark_price=safe_float(data.get("markPrice")),
            index_price=safe_float(data.get("indexPrice")),
            open_interest=safe_float(data.get("openInterest")),
            open_interest_notional=safe_float(data.get("openInterestValue")),
            funding_rate=safe_float(data.get("fundingRate")),
            volume_24h_base=safe_float(data.get("volume24h")),
            volume_24h_notional=safe_float(data.get("turnover24h")),
            timestamp_ms=safe_int(payload.get("ts")) or int(time.time() * 1000),
        )

    def _handle_bybit_liquidation(self, symbol: str, payload: Dict[str, object]) -> None:
        items = payload.get("data") or []
        if isinstance(items, dict):
            items = [items]
        with self.lock:
            for item in items:
                side = normalize_liquidation_side(item.get("side") or item.get("S"))
                price = safe_float(item.get("price")) or safe_float(item.get("p"))
                size = safe_float(item.get("size")) or safe_float(item.get("v"))
                event = LiquidationEvent(
                    exchange=self.clients["bybit"].exchange_name,
                    symbol=symbol,
                    timestamp_ms=safe_int(item.get("updatedTime")) or safe_int(item.get("T")) or int(time.time() * 1000),
                    side=side,
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="ws",
                    raw=item,
                )
                self._append_liquidation_event_locked("bybit", event)

    def _handle_bybit_orderbook(self, payload: Dict[str, object]) -> None:
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return
        bids = [(safe_float(price), safe_float(size)) for price, size in data.get("b", [])]
        asks = [(safe_float(price), safe_float(size)) for price, size in data.get("a", [])]
        with self.lock:
            if payload.get("type") == "snapshot" or not self.orderbooks["bybit"]:
                self._replace_orderbook_locked("bybit", bids, asks)
            else:
                self._apply_orderbook_delta_locked("bybit", bids, asks)

    def _handle_bybit_trade(self, symbol: str, payload: Dict[str, object]) -> None:
        items = payload.get("data") or []
        if isinstance(items, dict):
            items = [items]
        with self.lock:
            for item in items:
                price = safe_float(item.get("p")) or safe_float(item.get("price"))
                size = safe_float(item.get("v")) or safe_float(item.get("size"))
                event = TradeEvent(
                    exchange=self.clients["bybit"].exchange_name,
                    symbol=item.get("s") or symbol,
                    timestamp_ms=safe_int(item.get("T")) or safe_int(item.get("time")) or int(time.time() * 1000),
                    side=normalize_trade_side(item.get("S") or item.get("side")),
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="ws",
                    raw=item,
                )
                self._append_trade_event_locked("bybit", event)

    def _handle_binance_message(self, symbol: str, payload: Dict[str, object]) -> None:
        stream = str(payload.get("stream", ""))
        data = payload.get("data") or {}
        if "@forceorder" in stream.lower():
            order = data.get("o") or {}
            price = safe_float(order.get("ap")) or safe_float(order.get("p"))
            size = safe_float(order.get("z")) or safe_float(order.get("q"))
            event = LiquidationEvent(
                exchange=self.clients["binance"].exchange_name,
                symbol=order.get("s") or symbol,
                timestamp_ms=safe_int(data.get("E")) or safe_int(order.get("T")) or int(time.time() * 1000),
                side=normalize_liquidation_side(order.get("S")),
                price=price,
                size=size,
                notional=compute_notional(price, size),
                source="ws",
                raw=order,
            )
            with self.lock:
                self._append_liquidation_event_locked("binance", event)
            return

        if "@aggtrade" in stream.lower():
            price = safe_float(data.get("p"))
            size = safe_float(data.get("q"))
            event = TradeEvent(
                exchange=self.clients["binance"].exchange_name,
                symbol=data.get("s") or symbol,
                timestamp_ms=safe_int(data.get("E")) or safe_int(data.get("T")) or int(time.time() * 1000),
                side="sell" if bool(data.get("m")) else "buy",
                price=price,
                size=size,
                notional=compute_notional(price, size),
                source="ws",
                raw=data,
            )
            with self.lock:
                self._append_trade_event_locked("binance", event)
            return

        if "@depth@" in stream.lower():
            self._handle_binance_depth(symbol, data)
            return

        if "markprice" in stream.lower():
            self._update_live_snapshot(
                "binance",
                symbol,
                mark_price=safe_float(data.get("p")),
                index_price=safe_float(data.get("i")),
                funding_rate=safe_float(data.get("r")),
                timestamp_ms=safe_int(data.get("E")) or int(time.time() * 1000),
            )
        elif "@ticker" in stream.lower():
            self._update_live_snapshot(
                "binance",
                symbol,
                last_price=safe_float(data.get("c")),
                volume_24h_base=safe_float(data.get("v")),
                volume_24h_notional=safe_float(data.get("q")),
                timestamp_ms=safe_int(data.get("E")) or int(time.time() * 1000),
            )

    def _handle_binance_depth(self, symbol: str, data: Dict[str, object]) -> None:
        event = {
            "U": safe_int(data.get("U")),
            "u": safe_int(data.get("u")),
            "pu": safe_int(data.get("pu")),
            "bids": [(safe_float(price), safe_float(size)) for price, size in data.get("b", [])],
            "asks": [(safe_float(price), safe_float(size)) for price, size in data.get("a", [])],
        }
        if event["U"] is None or event["u"] is None:
            return

        should_bootstrap = False
        with self.lock:
            if not self.binance_depth_synced:
                self.binance_depth_buffer.append(event)
                if not self.binance_depth_bootstrapping:
                    self.binance_depth_bootstrapping = True
                    should_bootstrap = True
            else:
                if self.binance_depth_last_u is not None and event["pu"] not in (None, self.binance_depth_last_u):
                    self.binance_depth_synced = False
                    self.binance_depth_last_u = None
                    self.binance_depth_buffer.clear()
                    self.binance_depth_buffer.append(event)
                    if not self.binance_depth_bootstrapping:
                        self.binance_depth_bootstrapping = True
                        should_bootstrap = True
                else:
                    self._apply_orderbook_delta_locked("binance", event["bids"], event["asks"])
                    self.binance_depth_last_u = int(event["u"])

        if should_bootstrap:
            self._bootstrap_binance_depth(symbol)

    def _bootstrap_binance_depth(self, symbol: str) -> None:
        try:
            snapshot = fetch_binance_futures_orderbook_snapshot(symbol, 1000, timeout=self.timeout)
            last_update_id = safe_int(snapshot.get("lastUpdateId"))
            if last_update_id is None:
                raise ValueError("binance depth snapshot missing lastUpdateId")

            bids = [(safe_float(price), safe_float(size)) for price, size in snapshot.get("bids", [])]
            asks = [(safe_float(price), safe_float(size)) for price, size in snapshot.get("asks", [])]
            with self.lock:
                buffered_events = list(self.binance_depth_buffer)
                self.binance_depth_buffer.clear()
                self._replace_orderbook_locked("binance", bids, asks)
                previous_u = last_update_id
                for event in sorted(buffered_events, key=lambda item: (int(item["U"]), int(item["u"]))):
                    if int(event["u"]) < last_update_id:
                        continue
                    if int(event["U"]) > previous_u + 1:
                        self.binance_depth_synced = False
                        self.binance_depth_last_u = None
                        self.binance_depth_bootstrapping = False
                        return
                    self._apply_orderbook_delta_locked("binance", event["bids"], event["asks"])
                    previous_u = int(event["u"])

                self.binance_depth_last_u = previous_u
                self.binance_depth_synced = True
                self.binance_depth_bootstrapping = False
        except Exception as exc:
            with self.lock:
                self.binance_depth_synced = False
                self.binance_depth_last_u = None
                self.binance_depth_bootstrapping = False
            self._on_error("binance", symbol, exc)

    def _handle_okx_message(self, symbol: str, payload: Dict[str, object]) -> None:
        if payload.get("event"):
            return
        arg = payload.get("arg") or {}
        data_list = payload.get("data") or [{}]
        data = data_list[0]
        channel = arg.get("channel")
        if channel == "tickers":
            self._update_live_snapshot(
                "okx",
                symbol,
                last_price=safe_float(data.get("last")),
                volume_24h_base=safe_float(data.get("vol24h")),
                volume_24h_notional=safe_float(data.get("volCcy24h")),
                timestamp_ms=safe_int(data.get("ts")) or int(time.time() * 1000),
            )
        elif channel == "mark-price":
            self._update_live_snapshot(
                "okx",
                symbol,
                mark_price=safe_float(data.get("markPx")),
                timestamp_ms=safe_int(data.get("ts")) or int(time.time() * 1000),
            )
        elif channel == "books5":
            bids = [(safe_float(row[0]), safe_float(row[1])) for row in data.get("bids", [])]
            asks = [(safe_float(row[0]), safe_float(row[1])) for row in data.get("asks", [])]
            with self.lock:
                self._replace_orderbook_locked("okx", bids, asks)
        elif channel == "trades":
            with self.lock:
                for item in data_list:
                    price = safe_float(item.get("px"))
                    size = safe_float(item.get("sz"))
                    event = TradeEvent(
                        exchange=self.clients["okx"].exchange_name,
                        symbol=item.get("instId") or symbol,
                        timestamp_ms=safe_int(item.get("ts")) or int(time.time() * 1000),
                        side=normalize_trade_side(item.get("side")),
                        price=price,
                        size=size,
                        notional=compute_notional(price, size),
                        source="ws",
                        raw=item,
                    )
                    self._append_trade_event_locked("okx", event)

    def _handle_hyperliquid_message(self, symbol: str, payload: Dict[str, object]) -> None:
        channel = payload.get("channel")
        if channel == "allMids":
            mids = (payload.get("data") or {}).get("mids") or {}
            self._update_live_snapshot(
                "hyperliquid",
                symbol,
                last_price=safe_float(mids.get(symbol)),
                timestamp_ms=int(time.time() * 1000),
            )
        elif channel == "l2Book":
            data = payload.get("data") or {}
            levels = data.get("levels", [[], []])
            bids = [(safe_float(item.get("px")), safe_float(item.get("sz"))) for item in levels[0]]
            asks = [(safe_float(item.get("px")), safe_float(item.get("sz"))) for item in levels[1]]
            with self.lock:
                self._replace_orderbook_locked("hyperliquid", bids, asks)
        elif channel == "trades":
            items = payload.get("data") or []
            with self.lock:
                for item in items:
                    price = safe_float(item.get("px"))
                    size = safe_float(item.get("sz"))
                    event = TradeEvent(
                        exchange=self.clients["hyperliquid"].exchange_name,
                        symbol=item.get("coin") or symbol,
                        timestamp_ms=safe_int(item.get("time")) or int(time.time() * 1000),
                        side=normalize_trade_side(item.get("side")),
                        price=price,
                        size=size,
                        notional=compute_notional(price, size),
                        source="ws",
                        raw=item,
                    )
                    self._append_trade_event_locked("hyperliquid", event)

    def _handle_spot_message(self, exchange_key: str, symbol: str, payload: Dict[str, object]) -> None:
        if exchange_key == "binance":
            stream = str(payload.get("stream", ""))
            data = payload.get("data") or {}
            if "@aggtrade" in stream.lower():
                price = safe_float(data.get("p"))
                size = safe_float(data.get("q"))
                event = TradeEvent(
                    exchange=self.spot_clients["binance"].exchange_name,
                    symbol=data.get("s") or symbol,
                    timestamp_ms=safe_int(data.get("E")) or safe_int(data.get("T")) or int(time.time() * 1000),
                    side="sell" if bool(data.get("m")) else "buy",
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="ws",
                    raw=data,
                )
                with self.lock:
                    self._append_spot_trade_event_locked("binance", event)
                return
            if "@depth@" in stream.lower():
                self._handle_spot_depth("binance", symbol, data)
                return
            if "@ticker" in stream.lower():
                self._update_spot_snapshot(
                    "binance",
                    symbol,
                    last_price=safe_float(data.get("c")),
                    volume_24h_base=safe_float(data.get("v")),
                    volume_24h_notional=safe_float(data.get("q")),
                    timestamp_ms=safe_int(data.get("E")) or int(time.time() * 1000),
                )
            return

        if exchange_key == "bybit":
            topic = str(payload.get("topic", ""))
            if payload.get("success") is not None:
                return
            if topic.startswith("orderbook"):
                self._handle_bybit_spot_orderbook(payload)
            elif topic.startswith("publicTrade"):
                self._handle_bybit_spot_trade(symbol, payload)
            elif topic.startswith("tickers"):
                self._handle_bybit_spot_ticker(symbol, payload)
            return

        if exchange_key == "okx":
            self._handle_okx_spot_message(symbol, payload)

    def _handle_spot_depth(self, exchange_key: str, symbol: str, data: Dict[str, object]) -> None:
        event = {
            "U": safe_int(data.get("U")),
            "u": safe_int(data.get("u")),
            "bids": [(safe_float(price), safe_float(size)) for price, size in data.get("b", [])],
            "asks": [(safe_float(price), safe_float(size)) for price, size in data.get("a", [])],
        }
        if event["U"] is None or event["u"] is None:
            return
        should_bootstrap = False
        with self.lock:
            if not self.binance_spot_depth_synced:
                self.binance_spot_depth_buffer.append(event)
                if not self.binance_spot_depth_bootstrapping:
                    self.binance_spot_depth_bootstrapping = True
                    should_bootstrap = True
            else:
                if self.binance_spot_depth_last_u is not None and int(event["U"]) > self.binance_spot_depth_last_u + 1:
                    self.binance_spot_depth_synced = False
                    self.binance_spot_depth_last_u = None
                    self.binance_spot_depth_buffer.clear()
                    self.binance_spot_depth_buffer.append(event)
                    if not self.binance_spot_depth_bootstrapping:
                        self.binance_spot_depth_bootstrapping = True
                        should_bootstrap = True
                elif self.binance_spot_depth_last_u is None or int(event["u"]) > self.binance_spot_depth_last_u:
                    self._apply_spot_orderbook_delta_locked(exchange_key, event["bids"], event["asks"])
                    self.binance_spot_depth_last_u = int(event["u"])
        if should_bootstrap:
            self._bootstrap_spot_depth(exchange_key, symbol)

    def _bootstrap_spot_depth(self, exchange_key: str, symbol: str) -> None:
        try:
            snapshot = fetch_binance_spot_orderbook_snapshot(symbol, 1000, timeout=self.timeout)
            last_update_id = safe_int(snapshot.get("lastUpdateId"))
            if last_update_id is None:
                raise ValueError("binance spot depth snapshot missing lastUpdateId")
            bids = [(safe_float(price), safe_float(size)) for price, size in snapshot.get("bids", [])]
            asks = [(safe_float(price), safe_float(size)) for price, size in snapshot.get("asks", [])]
            with self.lock:
                buffered_events = list(self.binance_spot_depth_buffer)
                self.binance_spot_depth_buffer.clear()
                self._replace_spot_orderbook_locked(exchange_key, bids, asks)
                previous_u = last_update_id
                for event in sorted(buffered_events, key=lambda item: (int(item["U"]), int(item["u"]))):
                    if int(event["u"]) <= last_update_id:
                        continue
                    if int(event["U"]) > previous_u + 1:
                        self.binance_spot_depth_synced = False
                        self.binance_spot_depth_last_u = None
                        self.binance_spot_depth_bootstrapping = False
                        return
                    self._apply_spot_orderbook_delta_locked(exchange_key, event["bids"], event["asks"])
                    previous_u = int(event["u"])
                self.binance_spot_depth_last_u = previous_u
                self.binance_spot_depth_synced = True
                self.binance_spot_depth_bootstrapping = False
        except Exception as exc:
            with self.lock:
                self.binance_spot_depth_synced = False
                self.binance_spot_depth_last_u = None
                self.binance_spot_depth_bootstrapping = False
            self._on_spot_error(exchange_key, symbol, exc)

    def _handle_bybit_spot_ticker(self, symbol: str, payload: Dict[str, object]) -> None:
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return
        self._update_spot_snapshot(
            "bybit",
            symbol,
            last_price=safe_float(data.get("lastPrice")),
            bid_price=safe_float(data.get("bid1Price")),
            ask_price=safe_float(data.get("ask1Price")),
            volume_24h_base=safe_float(data.get("volume24h")),
            volume_24h_notional=safe_float(data.get("turnover24h")),
            timestamp_ms=safe_int(payload.get("ts")) or int(time.time() * 1000),
        )

    def _handle_bybit_spot_orderbook(self, payload: Dict[str, object]) -> None:
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return
        bids = [(safe_float(price), safe_float(size)) for price, size in data.get("b", [])]
        asks = [(safe_float(price), safe_float(size)) for price, size in data.get("a", [])]
        with self.lock:
            if payload.get("type") == "snapshot" or not self.spot_orderbooks["bybit"]:
                self._replace_spot_orderbook_locked("bybit", bids, asks)
            else:
                self._apply_spot_orderbook_delta_locked("bybit", bids, asks)

    def _handle_bybit_spot_trade(self, symbol: str, payload: Dict[str, object]) -> None:
        items = payload.get("data") or []
        if isinstance(items, dict):
            items = [items]
        with self.lock:
            for item in items:
                price = safe_float(item.get("p")) or safe_float(item.get("price"))
                size = safe_float(item.get("v")) or safe_float(item.get("size"))
                event = TradeEvent(
                    exchange=self.spot_clients["bybit"].exchange_name,
                    symbol=item.get("s") or symbol,
                    timestamp_ms=safe_int(item.get("T")) or safe_int(item.get("time")) or int(time.time() * 1000),
                    side=normalize_trade_side(item.get("S") or item.get("side")),
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="ws",
                    raw=item,
                )
                self._append_spot_trade_event_locked("bybit", event)

    def _handle_okx_spot_message(self, symbol: str, payload: Dict[str, object]) -> None:
        if payload.get("event"):
            return
        arg = payload.get("arg") or {}
        data_list = payload.get("data") or [{}]
        data = data_list[0]
        channel = arg.get("channel")
        if channel == "tickers":
            self._update_spot_snapshot(
                "okx",
                symbol,
                last_price=safe_float(data.get("last")),
                bid_price=safe_float(data.get("bidPx")),
                ask_price=safe_float(data.get("askPx")),
                volume_24h_base=safe_float(data.get("vol24h")),
                volume_24h_notional=safe_float(data.get("volCcy24h")),
                timestamp_ms=safe_int(data.get("ts")) or int(time.time() * 1000),
            )
        elif channel == "books5":
            bids = [(safe_float(row[0]), safe_float(row[1])) for row in data.get("bids", [])]
            asks = [(safe_float(row[0]), safe_float(row[1])) for row in data.get("asks", [])]
            with self.lock:
                self._replace_spot_orderbook_locked("okx", bids, asks)
        elif channel == "trades":
            with self.lock:
                for item in data_list:
                    price = safe_float(item.get("px"))
                    size = safe_float(item.get("sz"))
                    event = TradeEvent(
                        exchange=self.spot_clients["okx"].exchange_name,
                        symbol=item.get("instId") or symbol,
                        timestamp_ms=safe_int(item.get("ts")) or int(time.time() * 1000),
                        side=normalize_trade_side(item.get("side")),
                        price=price,
                        size=size,
                        notional=compute_notional(price, size),
                        source="ws",
                        raw=item,
                    )
                    self._append_spot_trade_event_locked("okx", event)
