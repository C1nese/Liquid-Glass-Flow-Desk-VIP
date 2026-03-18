from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from models import Candle, ExchangeSnapshot, LiquidationEvent, OIPoint, OrderBookLevel, SpotSnapshot, TradeEvent


DEFAULT_TIMEOUT = 10
EXCHANGE_ORDER = ("binance", "bybit", "okx", "hyperliquid")
SPOT_EXCHANGE_ORDER = ("binance", "bybit", "okx")
SUPPORTED_INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")

BYBIT_CANDLE_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}
BINANCE_CANDLE_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
OKX_CANDLE_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1Dutc",
}
HYPERLIQUID_CANDLE_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
BINANCE_OI_INTERVALS = {
    "1m": "5m",
    "3m": "5m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
BYBIT_OI_INTERVALS = {
    "1m": "5min",
    "3m": "5min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
BYBIT_RATIO_INTERVALS = {
    "1m": "5min",
    "3m": "5min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

EXCHANGE_TITLE_MAP = {
    "binance": "Binance",
    "bybit": "Bybit",
    "okx": "OKX",
    "hyperliquid": "Hyperliquid",
}

_REQUEST_HEALTH_LOCK = threading.Lock()
_REQUEST_HEALTH: Dict[str, Dict[str, Any]] = {}
_REQUEST_COOLDOWN_SECONDS = {
    "legal": 180,
    "forbidden": 90,
    "rate_limit": 45,
    "timeout": 20,
    "transport": 25,
    "server": 18,
    "http": 30,
    "other": 15,
}


def safe_float(value: Optional[object]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Optional[object]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_onchain_address(value: Optional[object]) -> str:
    return str(value or "").strip().lower()


def is_valid_onchain_address(value: Optional[object]) -> bool:
    text = normalize_onchain_address(value)
    if len(text) != 42 or not text.startswith("0x"):
        return False
    return all(char in "0123456789abcdef" for char in text[2:])


def interval_to_millis(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping.get(interval, 300_000)


def default_symbols(coin: str) -> Dict[str, str]:
    coin = coin.upper().strip()
    return {
        "bybit": f"{coin}USDT",
        "binance": f"{coin}USDT",
        "okx": f"{coin}-USDT-SWAP",
        "hyperliquid": coin,
    }


def default_spot_symbol(coin: str) -> str:
    return f"{coin.upper().strip()}USDT"


def default_spot_symbols(coin: str) -> Dict[str, str]:
    coin = coin.upper().strip()
    return {
        "binance": f"{coin}USDT",
        "bybit": f"{coin}USDT",
        "okx": f"{coin}-USDT",
    }


def _normalize_coin_code(value: Optional[object]) -> str:
    return str(value or "").strip().upper()


def _collect_bybit_instruments(client: "BaseClient", category: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cursor = ""
    for _ in range(8):
        params = {"category": category, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        payload = client._request("GET", "/v5/market/instruments-info", params=params)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        batch = result.get("list", []) if isinstance(result, dict) else []
        if isinstance(batch, list):
            rows.extend(item for item in batch if isinstance(item, dict))
        next_cursor = ""
        if isinstance(result, dict):
            next_cursor = str(result.get("nextPageCursor") or "").strip()
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    return rows


def fetch_exchange_coin_catalog(timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    availability: Dict[str, Dict[str, Dict[str, bool]]] = {}
    status: Dict[str, Dict[str, str]] = {
        exchange_key: {"perp": "unknown", "spot": "unknown"}
        for exchange_key in EXCHANGE_ORDER
    }
    errors: Dict[str, Dict[str, str]] = {
        exchange_key: {"perp": "", "spot": ""}
        for exchange_key in EXCHANGE_ORDER
    }
    status["hyperliquid"]["spot"] = "unsupported"

    def register(exchange_key: str, market: str, coins: List[str]) -> None:
        unique_coins = sorted({_normalize_coin_code(item) for item in coins if _normalize_coin_code(item)})
        for coin in unique_coins:
            coin_entry = availability.setdefault(
                coin,
                {
                    key: {"perp": False, "spot": False}
                    for key in EXCHANGE_ORDER
                },
            )
            coin_entry.setdefault(exchange_key, {"perp": False, "spot": False})[market] = True

    try:
        binance_perp_payload = BinanceClient(timeout)._request("GET", "/fapi/v1/exchangeInfo")
        register(
            "binance",
            "perp",
            [
                item.get("baseAsset")
                for item in binance_perp_payload.get("symbols", [])
                if str(item.get("quoteAsset") or "").upper() == "USDT"
                and str(item.get("contractType") or "").upper() == "PERPETUAL"
                and str(item.get("status") or "").upper() == "TRADING"
            ],
        )
        status["binance"]["perp"] = "ok"
    except Exception as exc:
        status["binance"]["perp"] = "error"
        errors["binance"]["perp"] = str(exc)

    try:
        binance_spot_payload = BinanceSpotClient(timeout)._request("GET", "/api/v3/exchangeInfo")
        register(
            "binance",
            "spot",
            [
                item.get("baseAsset")
                for item in binance_spot_payload.get("symbols", [])
                if str(item.get("quoteAsset") or "").upper() == "USDT"
                and str(item.get("status") or "").upper() == "TRADING"
            ],
        )
        status["binance"]["spot"] = "ok"
    except Exception as exc:
        status["binance"]["spot"] = "error"
        errors["binance"]["spot"] = str(exc)

    try:
        bybit_linear_rows = _collect_bybit_instruments(BybitClient(timeout), "linear")
        register(
            "bybit",
            "perp",
            [
                item.get("baseCoin")
                for item in bybit_linear_rows
                if str(item.get("quoteCoin") or "").upper() == "USDT"
                and str(item.get("status") or "").lower() in {"trading", "settling"}
            ],
        )
        status["bybit"]["perp"] = "ok"
    except Exception as exc:
        status["bybit"]["perp"] = "error"
        errors["bybit"]["perp"] = str(exc)

    try:
        bybit_spot_rows = _collect_bybit_instruments(BybitSpotClient(timeout), "spot")
        register(
            "bybit",
            "spot",
            [
                item.get("baseCoin")
                for item in bybit_spot_rows
                if str(item.get("quoteCoin") or "").upper() == "USDT"
                and str(item.get("status") or "").lower() in {"trading", "settling"}
            ],
        )
        status["bybit"]["spot"] = "ok"
    except Exception as exc:
        status["bybit"]["spot"] = "error"
        errors["bybit"]["spot"] = str(exc)

    try:
        okx_perp_payload = OkxClient(timeout)._request("GET", "/api/v5/public/instruments", params={"instType": "SWAP"})
        register(
            "okx",
            "perp",
            [
                str(item.get("instId") or "").split("-")[0]
                for item in okx_perp_payload.get("data", [])
                if "-USDT" in str(item.get("instId") or "").upper()
                and str(item.get("state") or "").lower() == "live"
            ],
        )
        status["okx"]["perp"] = "ok"
    except Exception as exc:
        status["okx"]["perp"] = "error"
        errors["okx"]["perp"] = str(exc)

    try:
        okx_spot_payload = OkxSpotClient(timeout)._request("GET", "/api/v5/public/instruments", params={"instType": "SPOT"})
        register(
            "okx",
            "spot",
            [
                item.get("baseCcy") or str(item.get("instId") or "").split("-")[0]
                for item in okx_spot_payload.get("data", [])
                if str(item.get("quoteCcy") or "").upper() == "USDT"
                and str(item.get("state") or "").lower() == "live"
            ],
        )
        status["okx"]["spot"] = "ok"
    except Exception as exc:
        status["okx"]["spot"] = "error"
        errors["okx"]["spot"] = str(exc)

    try:
        hyper_meta_payload = HyperliquidClient(timeout)._request("POST", "/info", json={"type": "meta"})
        universe = hyper_meta_payload.get("universe", []) if isinstance(hyper_meta_payload, dict) else []
        register(
            "hyperliquid",
            "perp",
            [item.get("name") for item in universe if isinstance(item, dict)],
        )
        status["hyperliquid"]["perp"] = "ok"
    except Exception as exc:
        status["hyperliquid"]["perp"] = "error"
        errors["hyperliquid"]["perp"] = str(exc)

    coins = sorted(availability)
    summary = {
        exchange_key: {
            "perp": sum(1 for coin in coins if availability.get(coin, {}).get(exchange_key, {}).get("perp")),
            "spot": sum(1 for coin in coins if availability.get(coin, {}).get(exchange_key, {}).get("spot")),
        }
        for exchange_key in EXCHANGE_ORDER
    }
    return {"coins": coins, "availability": availability, "summary": summary, "status": status, "errors": errors}


def compute_notional(price: Optional[float], size: Optional[float]) -> Optional[float]:
    if price is None or size is None:
        return None
    return price * size


def normalize_depth_limit(exchange_key: str, limit: int) -> int:
    if exchange_key == "binance":
        supported_limits = [5, 10, 20, 50, 100, 500, 1000]
        return min(supported_limits, key=lambda candidate: (abs(candidate - limit), candidate))
    return limit


def normalize_liquidation_side(value: Optional[object]) -> str:
    text = str(value or "").strip().lower()
    if text in {"long", "longs", "sell"}:
        return "long"
    if text in {"short", "shorts", "buy"}:
        return "short"
    return text or "unknown"


def normalize_trade_side(value: Optional[object]) -> str:
    text = str(value or "").strip().lower()
    if text in {"buy", "bid", "b"}:
        return "buy"
    if text in {"sell", "ask", "a", "s"}:
        return "sell"
    return text or "unknown"


def _request_health_key(exchange_name: str) -> Tuple[str, str, str]:
    normalized_name = str(exchange_name or "").strip()
    lower_name = normalized_name.lower()
    market = "spot" if "spot" in lower_name else "perp"
    if "binance" in lower_name:
        exchange_key = "binance"
    elif "bybit" in lower_name:
        exchange_key = "bybit"
    elif "okx" in lower_name:
        exchange_key = "okx"
    elif "hyperliquid" in lower_name:
        exchange_key = "hyperliquid"
    else:
        exchange_key = lower_name.replace(" ", "-") or "unknown"
    display_name = EXCHANGE_TITLE_MAP.get(exchange_key, normalized_name or exchange_key.title())
    return exchange_key, market, display_name


def _request_health_defaults(exchange_name: str) -> Dict[str, Any]:
    exchange_key, market, display_name = _request_health_key(exchange_name)
    return {
        "exchange_key": exchange_key,
        "exchange_name": display_name,
        "market": market,
        "last_success_ms": None,
        "last_attempt_ms": None,
        "last_error_ms": None,
        "last_error": None,
        "last_status_code": None,
        "error_kind": None,
        "consecutive_failures": 0,
        "cooldown_until_ms": None,
        "status": "idle",
    }


def _classify_request_error(exc: Exception) -> Tuple[Optional[int], str]:
    status_code = None
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status_code = int(exc.response.status_code)
    elif getattr(exc, "response", None) is not None and getattr(exc.response, "status_code", None) is not None:
        try:
            status_code = int(exc.response.status_code)
        except (TypeError, ValueError):
            status_code = None
    if status_code == 451:
        return status_code, "legal"
    if status_code == 403:
        return status_code, "forbidden"
    if status_code in (418, 429):
        return status_code, "rate_limit"
    if status_code is not None and status_code >= 500:
        return status_code, "server"
    if status_code is not None:
        return status_code, "http"
    if isinstance(exc, requests.Timeout):
        return None, "timeout"
    if isinstance(exc, requests.RequestException):
        return None, "transport"
    return None, "other"


def _current_cooldown_remaining_ms(exchange_name: str) -> int:
    exchange_key, market, _ = _request_health_key(exchange_name)
    cache_key = f"{exchange_key}:{market}"
    now_ms = int(time.time() * 1000)
    with _REQUEST_HEALTH_LOCK:
        entry = _REQUEST_HEALTH.get(cache_key)
        cooldown_until_ms = int(entry.get("cooldown_until_ms") or 0) if entry else 0
    return max(0, cooldown_until_ms - now_ms)


def _record_request_success(exchange_name: str) -> None:
    exchange_key, market, display_name = _request_health_key(exchange_name)
    cache_key = f"{exchange_key}:{market}"
    now_ms = int(time.time() * 1000)
    with _REQUEST_HEALTH_LOCK:
        entry = dict(_REQUEST_HEALTH.get(cache_key) or _request_health_defaults(exchange_name))
        entry.update(
            {
                "exchange_key": exchange_key,
                "exchange_name": display_name,
                "market": market,
                "last_success_ms": now_ms,
                "last_attempt_ms": now_ms,
                "consecutive_failures": 0,
                "cooldown_until_ms": None,
                "status": "ok",
            }
        )
        _REQUEST_HEALTH[cache_key] = entry


def _record_request_failure(exchange_name: str, exc: Exception) -> None:
    exchange_key, market, display_name = _request_health_key(exchange_name)
    cache_key = f"{exchange_key}:{market}"
    now_ms = int(time.time() * 1000)
    status_code, error_kind = _classify_request_error(exc)
    cooldown_seconds = int(_REQUEST_COOLDOWN_SECONDS.get(error_kind, _REQUEST_COOLDOWN_SECONDS["other"]))
    with _REQUEST_HEALTH_LOCK:
        entry = dict(_REQUEST_HEALTH.get(cache_key) or _request_health_defaults(exchange_name))
        consecutive_failures = int(entry.get("consecutive_failures") or 0) + 1
        should_cooldown = error_kind in {"legal", "forbidden", "rate_limit"} or consecutive_failures >= 2
        cooldown_until_ms = now_ms + cooldown_seconds * 1000 if should_cooldown else None
        entry.update(
            {
                "exchange_key": exchange_key,
                "exchange_name": display_name,
                "market": market,
                "last_attempt_ms": now_ms,
                "last_error_ms": now_ms,
                "last_error": str(exc),
                "last_status_code": status_code,
                "error_kind": error_kind,
                "consecutive_failures": consecutive_failures,
                "cooldown_until_ms": cooldown_until_ms,
                "status": "error",
            }
        )
        _REQUEST_HEALTH[cache_key] = entry


def describe_exchange_request_health() -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    rows: List[Dict[str, Any]] = []
    market_matrix = {
        "binance": ("perp", "spot"),
        "bybit": ("perp", "spot"),
        "okx": ("perp", "spot"),
        "hyperliquid": ("perp",),
    }
    with _REQUEST_HEALTH_LOCK:
        snapshot = {key: dict(value) for key, value in _REQUEST_HEALTH.items()}
    for exchange_key in EXCHANGE_ORDER:
        for market in market_matrix.get(exchange_key, ("perp",)):
            cache_key = f"{exchange_key}:{market}"
            entry = snapshot.get(cache_key, {})
            display_name = EXCHANGE_TITLE_MAP.get(exchange_key, exchange_key.title())
            cooldown_until_ms = int(entry.get("cooldown_until_ms") or 0)
            rows.append(
                {
                    "exchange_key": exchange_key,
                    "exchange_name": display_name,
                    "market": market,
                    "last_success_ms": entry.get("last_success_ms"),
                    "last_attempt_ms": entry.get("last_attempt_ms"),
                    "last_error_ms": entry.get("last_error_ms"),
                    "last_error": entry.get("last_error"),
                    "last_status_code": entry.get("last_status_code"),
                    "error_kind": entry.get("error_kind"),
                    "consecutive_failures": int(entry.get("consecutive_failures") or 0),
                    "cooldown_until_ms": cooldown_until_ms or None,
                    "cooldown_remaining_s": max(0, int((cooldown_until_ms - now_ms + 999) // 1000)) if cooldown_until_ms else 0,
                    "status": entry.get("status") or "idle",
                }
            )
    return rows


class BaseClient:
    exchange_name = "Unknown"
    base_url = ""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "exchange-liquidity-gui/2.0"})
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.35,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request(self, method: str, path: str, *, params=None, json=None):
        cooldown_remaining_ms = _current_cooldown_remaining_ms(self.exchange_name)
        if cooldown_remaining_ms > 0:
            remaining_seconds = max(1, int((cooldown_remaining_ms + 999) // 1000))
            raise RuntimeError(f"{self.exchange_name} REST 冷却中，{remaining_seconds}s 后自动重试")
        try:
            response = self.session.request(
                method,
                self.base_url + path,
                params=params,
                json=json,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            _record_request_failure(self.exchange_name, exc)
            raise
        _record_request_success(self.exchange_name)
        return payload

    def fetch(self, symbol: str) -> ExchangeSnapshot:
        raise NotImplementedError

    def fetch_candles(self, symbol: str, interval: str, limit: int) -> List[Candle]:
        raise NotImplementedError

    def fetch_orderbook(self, symbol: str, limit: int) -> List[OrderBookLevel]:
        raise NotImplementedError

    def fetch_open_interest_history(self, symbol: str, interval: str, limit: int) -> List[OIPoint]:
        return []

    def fetch_liquidations(self, symbol: str, limit: int) -> List[LiquidationEvent]:
        return []

    def fetch_trades(self, symbol: str, limit: int) -> List[TradeEvent]:
        return []

    def _error(self, symbol: str, exc: Exception) -> ExchangeSnapshot:
        return ExchangeSnapshot(
            exchange=self.exchange_name,
            symbol=symbol,
            status="error",
            error=str(exc),
        )


class BybitClient(BaseClient):
    exchange_name = "Bybit"
    base_url = "https://api.bybit.com"

    def fetch(self, symbol: str) -> ExchangeSnapshot:
        try:
            payload = self._request(
                "GET",
                "/v5/market/tickers",
                params={"category": "linear", "symbol": symbol},
            )
            items = payload.get("result", {}).get("list", [])
            if not items:
                raise ValueError("empty ticker response")
            item = items[0]
            return ExchangeSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                last_price=safe_float(item.get("lastPrice")),
                mark_price=safe_float(item.get("markPrice")),
                index_price=safe_float(item.get("indexPrice")),
                open_interest=safe_float(item.get("openInterest")),
                open_interest_notional=safe_float(item.get("openInterestValue")),
                funding_rate=safe_float(item.get("fundingRate")),
                volume_24h_base=safe_float(item.get("volume24h")),
                volume_24h_notional=safe_float(item.get("turnover24h")),
                timestamp_ms=safe_int(payload.get("time")),
                raw=item,
            )
        except Exception as exc:
            return self._error(symbol, exc)

    def fetch_candles(self, symbol: str, interval: str, limit: int) -> List[Candle]:
        payload = self._request(
            "GET",
            "/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": BYBIT_CANDLE_INTERVALS.get(interval, "5"),
                "limit": max(10, min(limit, 1000)),
            },
        )
        items = payload.get("result", {}).get("list", [])
        candles: List[Candle] = []
        for row in reversed(items):
            candles.append(
                Candle(
                    timestamp_ms=safe_int(row[0]) or 0,
                    open=safe_float(row[1]) or 0.0,
                    high=safe_float(row[2]) or 0.0,
                    low=safe_float(row[3]) or 0.0,
                    close=safe_float(row[4]) or 0.0,
                    volume=safe_float(row[5]) or 0.0,
                )
            )
        return candles

    def fetch_orderbook(self, symbol: str, limit: int) -> List[OrderBookLevel]:
        payload = self._request(
            "GET",
            "/v5/market/orderbook",
            params={"category": "linear", "symbol": symbol, "limit": max(1, min(limit, 200))},
        )
        result = payload.get("result", {})
        levels: List[OrderBookLevel] = []
        for price, size in result.get("b", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="bid"))
        for price, size in result.get("a", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="ask"))
        return levels

    def fetch_open_interest_history(self, symbol: str, interval: str, limit: int) -> List[OIPoint]:
        payload = self._request(
            "GET",
            "/v5/market/open-interest",
            params={
                "category": "linear",
                "symbol": symbol,
                "intervalTime": BYBIT_OI_INTERVALS.get(interval, "5min"),
                "limit": max(10, min(limit, 200)),
            },
        )
        items = payload.get("result", {}).get("list", [])
        points: List[OIPoint] = []
        for item in reversed(items):
            points.append(
                OIPoint(
                    timestamp_ms=safe_int(item.get("timestamp")) or 0,
                    open_interest=safe_float(item.get("openInterest")),
                    open_interest_notional=None,
                )
            )
        return points

    def fetch_trades(self, symbol: str, limit: int) -> List[TradeEvent]:
        payload = self._request(
            "GET",
            "/v5/market/recent-trade",
            params={"category": "linear", "symbol": symbol, "limit": max(10, min(limit, 1000))},
        )
        items = payload.get("result", {}).get("list", [])
        events: List[TradeEvent] = []
        for item in items:
            price = safe_float(item.get("price"))
            size = safe_float(item.get("size"))
            events.append(
                TradeEvent(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    timestamp_ms=safe_int(item.get("time")) or 0,
                    side=normalize_trade_side(item.get("side")),
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="rest",
                    raw=item,
                )
            )
        return events


class BinanceClient(BaseClient):
    exchange_name = "Binance"
    base_url = "https://fapi.binance.com"

    def fetch(self, symbol: str) -> ExchangeSnapshot:
        try:
            stats = self._request("GET", "/fapi/v1/ticker/24hr", params={"symbol": symbol})
            premium = self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})
            open_interest_payload = self._request("GET", "/fapi/v1/openInterest", params={"symbol": symbol})

            last_price = safe_float(stats.get("lastPrice"))
            mark_price = safe_float(premium.get("markPrice"))
            open_interest = safe_float(open_interest_payload.get("openInterest"))
            open_interest_notional = None
            if open_interest is not None and mark_price is not None:
                open_interest_notional = open_interest * mark_price

            return ExchangeSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                last_price=last_price,
                mark_price=mark_price,
                index_price=safe_float(premium.get("indexPrice")),
                open_interest=open_interest,
                open_interest_notional=open_interest_notional,
                funding_rate=safe_float(premium.get("lastFundingRate")),
                volume_24h_base=safe_float(stats.get("volume")),
                volume_24h_notional=safe_float(stats.get("quoteVolume")),
                timestamp_ms=safe_int(stats.get("closeTime")),
                raw={
                    "ticker_24h": stats,
                    "premium_index": premium,
                    "open_interest": open_interest_payload,
                },
            )
        except Exception as exc:
            return self._error(symbol, exc)

    def fetch_candles(self, symbol: str, interval: str, limit: int) -> List[Candle]:
        payload = self._request(
            "GET",
            "/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": BINANCE_CANDLE_INTERVALS.get(interval, "5m"),
                "limit": max(10, min(limit, 1500)),
            },
        )
        candles: List[Candle] = []
        for row in payload:
            candles.append(
                Candle(
                    timestamp_ms=safe_int(row[0]) or 0,
                    open=safe_float(row[1]) or 0.0,
                    high=safe_float(row[2]) or 0.0,
                    low=safe_float(row[3]) or 0.0,
                    close=safe_float(row[4]) or 0.0,
                    volume=safe_float(row[5]) or 0.0,
                )
            )
        return candles

    def fetch_orderbook(self, symbol: str, limit: int) -> List[OrderBookLevel]:
        normalized_limit = normalize_depth_limit("binance", max(5, min(limit, 1000)))
        payload = self._request(
            "GET",
            "/fapi/v1/depth",
            params={"symbol": symbol, "limit": normalized_limit},
        )
        levels: List[OrderBookLevel] = []
        for price, size in payload.get("bids", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="bid"))
        for price, size in payload.get("asks", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="ask"))
        return levels

    def fetch_open_interest_history(self, symbol: str, interval: str, limit: int) -> List[OIPoint]:
        payload = self._request(
            "GET",
            "/futures/data/openInterestHist",
            params={
                "symbol": symbol,
                "period": BINANCE_OI_INTERVALS.get(interval, "5m"),
                "limit": max(10, min(limit, 500)),
            },
        )
        points: List[OIPoint] = []
        for item in payload:
            points.append(
                OIPoint(
                    timestamp_ms=safe_int(item.get("timestamp")) or 0,
                    open_interest=safe_float(item.get("sumOpenInterest")),
                    open_interest_notional=safe_float(item.get("sumOpenInterestValue")),
                )
            )
        return points

    def fetch_trades(self, symbol: str, limit: int) -> List[TradeEvent]:
        payload = self._request(
            "GET",
            "/fapi/v1/aggTrades",
            params={"symbol": symbol, "limit": max(10, min(limit, 1000))},
        )
        events: List[TradeEvent] = []
        for item in payload:
            price = safe_float(item.get("p"))
            size = safe_float(item.get("q"))
            events.append(
                TradeEvent(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    timestamp_ms=safe_int(item.get("T")) or 0,
                    side="sell" if bool(item.get("m")) else "buy",
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="rest",
                    raw=item,
                )
            )
        return events

    def fetch_liquidations(self, symbol: str, limit: int) -> List[LiquidationEvent]:
        payload = self._request(
            "GET",
            "/fapi/v1/allForceOrders",
            params={"symbol": symbol, "limit": max(10, min(limit, 100))},
        )
        items = payload if isinstance(payload, list) else payload.get("data", [])
        events: List[LiquidationEvent] = []
        for item in items:
            price = safe_float(item.get("avgPrice")) or safe_float(item.get("averagePrice")) or safe_float(item.get("price"))
            size = safe_float(item.get("executedQty")) or safe_float(item.get("origQty"))
            notional = compute_notional(price, size) or safe_float(item.get("cumQuote"))
            events.append(
                LiquidationEvent(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    timestamp_ms=safe_int(item.get("time")) or safe_int(item.get("updatedTime")) or 0,
                    side=normalize_liquidation_side(item.get("side")),
                    price=price,
                    size=size,
                    notional=notional,
                    source="rest",
                    raw=item,
                )
            )
        return events


class BinanceSpotClient(BaseClient):
    exchange_name = "Binance Spot"
    base_url = "https://api.binance.com"

    def fetch(self, symbol: str) -> SpotSnapshot:
        try:
            stats = self._request("GET", "/api/v3/ticker/24hr", params={"symbol": symbol})
            book = self._request("GET", "/api/v3/ticker/bookTicker", params={"symbol": symbol})
            return SpotSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                last_price=safe_float(stats.get("lastPrice")),
                bid_price=safe_float(book.get("bidPrice")),
                ask_price=safe_float(book.get("askPrice")),
                volume_24h_base=safe_float(stats.get("volume")),
                volume_24h_notional=safe_float(stats.get("quoteVolume")),
                timestamp_ms=safe_int(stats.get("closeTime")),
                raw={"ticker_24h": stats, "book_ticker": book},
            )
        except Exception as exc:
            return SpotSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                status="error",
                error=str(exc),
            )

    def fetch_orderbook(self, symbol: str, limit: int) -> List[OrderBookLevel]:
        normalized_limit = normalize_depth_limit("binance", max(5, min(limit, 1000)))
        payload = self._request(
            "GET",
            "/api/v3/depth",
            params={"symbol": symbol, "limit": normalized_limit},
        )
        levels: List[OrderBookLevel] = []
        for price, size in payload.get("bids", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="bid"))
        for price, size in payload.get("asks", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="ask"))
        return levels

    def fetch_trades(self, symbol: str, limit: int) -> List[TradeEvent]:
        payload = self._request(
            "GET",
            "/api/v3/trades",
            params={"symbol": symbol, "limit": max(10, min(limit, 1000))},
        )
        events: List[TradeEvent] = []
        for item in payload:
            price = safe_float(item.get("price"))
            size = safe_float(item.get("qty"))
            events.append(
                TradeEvent(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    timestamp_ms=safe_int(item.get("time")) or 0,
                    side="sell" if bool(item.get("isBuyerMaker")) else "buy",
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="rest",
                    raw=item,
                )
            )
        return events


class BybitSpotClient(BaseClient):
    exchange_name = "Bybit Spot"
    base_url = "https://api.bybit.com"

    def fetch(self, symbol: str) -> SpotSnapshot:
        try:
            payload = self._request("GET", "/v5/market/tickers", params={"category": "spot", "symbol": symbol})
            items = payload.get("result", {}).get("list", [])
            if not items:
                raise ValueError("empty ticker response")
            item = items[0]
            return SpotSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                last_price=safe_float(item.get("lastPrice")),
                bid_price=safe_float(item.get("bid1Price")),
                ask_price=safe_float(item.get("ask1Price")),
                volume_24h_base=safe_float(item.get("volume24h")),
                volume_24h_notional=safe_float(item.get("turnover24h")),
                timestamp_ms=safe_int(payload.get("time")),
                raw=item,
            )
        except Exception as exc:
            return SpotSnapshot(exchange=self.exchange_name, symbol=symbol, status="error", error=str(exc))

    def fetch_orderbook(self, symbol: str, limit: int) -> List[OrderBookLevel]:
        payload = self._request(
            "GET",
            "/v5/market/orderbook",
            params={"category": "spot", "symbol": symbol, "limit": max(1, min(limit, 200))},
        )
        result = payload.get("result", {})
        levels: List[OrderBookLevel] = []
        for price, size in result.get("b", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="bid"))
        for price, size in result.get("a", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="ask"))
        return levels

    def fetch_trades(self, symbol: str, limit: int) -> List[TradeEvent]:
        payload = self._request(
            "GET",
            "/v5/market/recent-trade",
            params={"category": "spot", "symbol": symbol, "limit": max(10, min(limit, 1000))},
        )
        items = payload.get("result", {}).get("list", [])
        events: List[TradeEvent] = []
        for item in items:
            price = safe_float(item.get("price"))
            size = safe_float(item.get("size"))
            events.append(
                TradeEvent(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    timestamp_ms=safe_int(item.get("time")) or 0,
                    side=normalize_trade_side(item.get("side")),
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="rest",
                    raw=item,
                )
            )
        return events


class OkxSpotClient(BaseClient):
    exchange_name = "OKX Spot"
    base_url = "https://www.okx.com"

    def fetch(self, symbol: str) -> SpotSnapshot:
        try:
            payload = self._request("GET", "/api/v5/market/ticker", params={"instId": symbol})
            item = (payload.get("data") or [{}])[0]
            return SpotSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                last_price=safe_float(item.get("last")),
                bid_price=safe_float(item.get("bidPx")),
                ask_price=safe_float(item.get("askPx")),
                volume_24h_base=safe_float(item.get("vol24h")),
                volume_24h_notional=safe_float(item.get("volCcy24h")),
                timestamp_ms=safe_int(item.get("ts")),
                raw=item,
            )
        except Exception as exc:
            return SpotSnapshot(exchange=self.exchange_name, symbol=symbol, status="error", error=str(exc))

    def fetch_orderbook(self, symbol: str, limit: int) -> List[OrderBookLevel]:
        payload = self._request(
            "GET",
            "/api/v5/market/books",
            params={"instId": symbol, "sz": max(1, min(limit, 400))},
        )
        item = (payload.get("data") or [{}])[0]
        levels: List[OrderBookLevel] = []
        for price, size, *_ in item.get("bids", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="bid"))
        for price, size, *_ in item.get("asks", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="ask"))
        return levels

    def fetch_trades(self, symbol: str, limit: int) -> List[TradeEvent]:
        payload = self._request(
            "GET",
            "/api/v5/market/trades",
            params={"instId": symbol, "limit": max(10, min(limit, 100))},
        )
        items = payload.get("data", [])
        events: List[TradeEvent] = []
        for item in items:
            price = safe_float(item.get("px"))
            size = safe_float(item.get("sz"))
            events.append(
                TradeEvent(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    timestamp_ms=safe_int(item.get("ts")) or 0,
                    side=normalize_trade_side(item.get("side")),
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="rest",
                    raw=item,
                )
            )
        return events


class OkxClient(BaseClient):
    exchange_name = "OKX"
    base_url = "https://www.okx.com"

    def fetch(self, symbol: str) -> ExchangeSnapshot:
        try:
            ticker_payload = self._request("GET", "/api/v5/market/ticker", params={"instId": symbol})
            mark_payload = self._request(
                "GET",
                "/api/v5/public/mark-price",
                params={"instType": "SWAP", "instId": symbol},
            )
            oi_payload = self._request(
                "GET",
                "/api/v5/public/open-interest",
                params={"instType": "SWAP", "instId": symbol},
            )
            funding_payload = self._request(
                "GET",
                "/api/v5/public/funding-rate",
                params={"instId": symbol},
            )

            ticker = (ticker_payload.get("data") or [{}])[0]
            mark = (mark_payload.get("data") or [{}])[0]
            oi_item = (oi_payload.get("data") or [{}])[0]
            funding = (funding_payload.get("data") or [{}])[0]

            open_interest = safe_float(oi_item.get("oi"))
            open_interest_notional = safe_float(oi_item.get("oiUsd"))
            mark_price = safe_float(mark.get("markPx"))
            if open_interest_notional is None and open_interest is not None and mark_price is not None:
                open_interest_notional = open_interest * mark_price

            return ExchangeSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                last_price=safe_float(ticker.get("last")),
                mark_price=mark_price,
                index_price=None,
                open_interest=open_interest,
                open_interest_notional=open_interest_notional,
                funding_rate=safe_float(funding.get("fundingRate")),
                volume_24h_base=safe_float(ticker.get("vol24h")),
                volume_24h_notional=safe_float(ticker.get("volCcy24h")),
                timestamp_ms=safe_int(ticker.get("ts")),
                raw={
                    "ticker": ticker,
                    "mark_price": mark,
                    "open_interest": oi_item,
                    "funding_rate": funding,
                },
            )
        except Exception as exc:
            return self._error(symbol, exc)

    def fetch_candles(self, symbol: str, interval: str, limit: int) -> List[Candle]:
        payload = self._request(
            "GET",
            "/api/v5/market/candles",
            params={
                "instId": symbol,
                "bar": OKX_CANDLE_INTERVALS.get(interval, "5m"),
                "limit": max(10, min(limit, 300)),
            },
        )
        items = payload.get("data", [])
        candles: List[Candle] = []
        for row in reversed(items):
            candles.append(
                Candle(
                    timestamp_ms=safe_int(row[0]) or 0,
                    open=safe_float(row[1]) or 0.0,
                    high=safe_float(row[2]) or 0.0,
                    low=safe_float(row[3]) or 0.0,
                    close=safe_float(row[4]) or 0.0,
                    volume=safe_float(row[5]) or 0.0,
                )
            )
        return candles

    def fetch_orderbook(self, symbol: str, limit: int) -> List[OrderBookLevel]:
        payload = self._request(
            "GET",
            "/api/v5/market/books",
            params={"instId": symbol, "sz": max(1, min(limit, 400))},
        )
        data = (payload.get("data") or [{}])[0]
        levels: List[OrderBookLevel] = []
        for price, size, *_ in data.get("bids", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="bid"))
        for price, size, *_ in data.get("asks", []):
            levels.append(OrderBookLevel(price=safe_float(price) or 0.0, size=safe_float(size) or 0.0, side="ask"))
        return levels

    def fetch_trades(self, symbol: str, limit: int) -> List[TradeEvent]:
        payload = self._request(
            "GET",
            "/api/v5/market/trades",
            params={"instId": symbol, "limit": max(10, min(limit, 100))},
        )
        items = payload.get("data", [])
        events: List[TradeEvent] = []
        for item in items:
            price = safe_float(item.get("px"))
            size = safe_float(item.get("sz"))
            events.append(
                TradeEvent(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    timestamp_ms=safe_int(item.get("ts")) or 0,
                    side=normalize_trade_side(item.get("side")),
                    price=price,
                    size=size,
                    notional=compute_notional(price, size),
                    source="rest",
                    raw=item,
                )
            )
        return events


class HyperliquidClient(BaseClient):
    exchange_name = "Hyperliquid"
    base_url = "https://api.hyperliquid.xyz"

    def fetch(self, symbol: str) -> ExchangeSnapshot:
        try:
            payload = self._request("POST", "/info", json={"type": "metaAndAssetCtxs"})
            if not isinstance(payload, list) or len(payload) != 2:
                raise ValueError("unexpected metaAndAssetCtxs response")

            meta = payload[0]
            asset_contexts = payload[1]
            universe = meta.get("universe", [])

            asset_index = None
            for index, asset in enumerate(universe):
                if asset.get("name") == symbol:
                    asset_index = index
                    break

            if asset_index is None:
                raise ValueError(f"symbol {symbol} not found in Hyperliquid universe")

            ctx = asset_contexts[asset_index]
            mark_price = safe_float(ctx.get("markPx"))
            open_interest = safe_float(ctx.get("openInterest"))
            open_interest_notional = None
            if open_interest is not None and mark_price is not None:
                open_interest_notional = open_interest * mark_price

            last_price = safe_float(ctx.get("midPx")) or mark_price

            return ExchangeSnapshot(
                exchange=self.exchange_name,
                symbol=symbol,
                last_price=last_price,
                mark_price=mark_price,
                index_price=safe_float(ctx.get("oraclePx")),
                open_interest=open_interest,
                open_interest_notional=open_interest_notional,
                funding_rate=safe_float(ctx.get("funding")),
                volume_24h_base=safe_float(ctx.get("dayBaseVlm")),
                volume_24h_notional=safe_float(ctx.get("dayNtlVlm")),
                timestamp_ms=int(time.time() * 1000),
                raw={"meta": meta, "asset_context": ctx},
            )
        except Exception as exc:
            return self._error(symbol, exc)

    def fetch_candles(self, symbol: str, interval: str, limit: int) -> List[Candle]:
        interval_ms = interval_to_millis(interval)
        end_time = int(time.time() * 1000)
        start_time = max(0, end_time - interval_ms * (limit + 10))
        payload = self._request(
            "POST",
            "/info",
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": HYPERLIQUID_CANDLE_INTERVALS.get(interval, "5m"),
                    "startTime": start_time,
                    "endTime": end_time,
                },
            },
        )
        candles: List[Candle] = []
        for row in payload[-limit:]:
            candles.append(
                Candle(
                    timestamp_ms=safe_int(row.get("t")) or 0,
                    open=safe_float(row.get("o")) or 0.0,
                    high=safe_float(row.get("h")) or 0.0,
                    low=safe_float(row.get("l")) or 0.0,
                    close=safe_float(row.get("c")) or 0.0,
                    volume=safe_float(row.get("v")) or 0.0,
                )
            )
        return candles

    def fetch_orderbook(self, symbol: str, limit: int) -> List[OrderBookLevel]:
        payload = self._request("POST", "/info", json={"type": "l2Book", "coin": symbol})
        levels: List[OrderBookLevel] = []
        book_levels = payload.get("levels", [[], []])
        for row in book_levels[0][:limit]:
            levels.append(OrderBookLevel(price=safe_float(row.get("px")) or 0.0, size=safe_float(row.get("sz")) or 0.0, side="bid"))
        for row in book_levels[1][:limit]:
            levels.append(OrderBookLevel(price=safe_float(row.get("px")) or 0.0, size=safe_float(row.get("sz")) or 0.0, side="ask"))
        return levels

    def fetch_clearinghouse_state(self, address: str) -> Dict[str, Any]:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        payload = self._request(
            "POST",
            "/info",
            json={"type": "clearinghouseState", "user": normalized_address},
        )
        return payload if isinstance(payload, dict) else {}

    def fetch_user_funding(self, address: str, start_time_ms: int, end_time_ms: Optional[int] = None) -> List[dict]:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        payload = self._request(
            "POST",
            "/info",
            json={
                "type": "userFunding",
                "user": normalized_address,
                "startTime": int(start_time_ms),
                "endTime": int(end_time_ms or int(time.time() * 1000)),
            },
        )
        return payload if isinstance(payload, list) else []

    def fetch_user_fills(self, address: str) -> List[dict]:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        payload = self._request(
            "POST",
            "/info",
            json={"type": "userFills", "user": normalized_address},
        )
        return payload if isinstance(payload, list) else []

    def fetch_active_asset_data(self, address: str, symbol: str) -> Dict[str, Any]:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        payload = self._request(
            "POST",
            "/info",
            json={"type": "activeAssetData", "user": normalized_address, "coin": symbol.upper().strip()},
        )
        return payload if isinstance(payload, dict) else {}

    def fetch_user_role(self, address: str) -> Dict[str, Any]:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        payload = self._request(
            "POST",
            "/info",
            json={"type": "userRole", "user": normalized_address},
        )
        return payload if isinstance(payload, dict) else {}

    def fetch_portfolio(self, address: str) -> List[dict]:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        payload = self._request(
            "POST",
            "/info",
            json={"type": "portfolio", "user": normalized_address},
        )
        return payload if isinstance(payload, list) else []

    def fetch_user_vault_equities(self, address: str) -> List[dict]:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        payload = self._request(
            "POST",
            "/info",
            json={"type": "userVaultEquities", "user": normalized_address},
        )
        return payload if isinstance(payload, list) else []

    def fetch_vault_details(self, vault_address: str, user: Optional[str] = None) -> Dict[str, Any]:
        normalized_vault_address = normalize_onchain_address(vault_address)
        if not is_valid_onchain_address(normalized_vault_address):
            raise ValueError("invalid Hyperliquid vault address")
        body: Dict[str, Any] = {"type": "vaultDetails", "vaultAddress": normalized_vault_address}
        if user:
            normalized_user = normalize_onchain_address(user)
            if is_valid_onchain_address(normalized_user):
                body["user"] = normalized_user
        payload = self._request("POST", "/info", json=body)
        return payload if isinstance(payload, dict) else {}

    def fetch_spot_meta_and_asset_contexts(self) -> Dict[str, Any]:
        payload = self._request("POST", "/info", json={"type": "spotMetaAndAssetCtxs"})
        if isinstance(payload, list) and len(payload) == 2:
            meta = payload[0] if isinstance(payload[0], dict) else {}
            contexts = payload[1] if isinstance(payload[1], list) else []
            return {"meta": meta, "contexts": contexts}
        return {"meta": {}, "contexts": []}

    def fetch_spot_clearinghouse_state(self, address: str) -> Dict[str, Any]:
        normalized_address = normalize_onchain_address(address)
        if not is_valid_onchain_address(normalized_address):
            raise ValueError("invalid Hyperliquid address")
        payload = self._request(
            "POST",
            "/info",
            json={"type": "spotClearinghouseState", "user": normalized_address},
        )
        return payload if isinstance(payload, dict) else {}

    def fetch_perps_at_open_interest_cap(self, dex: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"type": "perpsAtOpenInterestCap"}
        if dex:
            body["dex"] = str(dex)
        payload = self._request("POST", "/info", json=body)
        return payload if isinstance(payload, dict) else {}


def _parse_hyperliquid_fill(item: dict) -> Dict[str, Any]:
    fill = item.get("fill") if isinstance(item.get("fill"), dict) else item
    price = safe_float(fill.get("px")) or safe_float(fill.get("price"))
    size = safe_float(fill.get("sz")) or safe_float(fill.get("size"))
    return {
        "time": safe_int(fill.get("time")) or 0,
        "coin": str(fill.get("coin") or ""),
        "direction": str(fill.get("dir") or fill.get("side") or ""),
        "side": normalize_trade_side(fill.get("side")),
        "price": price,
        "size": size,
        "notional": compute_notional(price, size),
        "closed_pnl": safe_float(fill.get("closedPnl")),
        "fee": safe_float(fill.get("fee")),
        "fee_token": str(fill.get("feeToken") or ""),
        "start_position": safe_float(fill.get("startPosition")),
        "hash": str(fill.get("hash") or ""),
        "raw": fill,
    }


def _parse_hyperliquid_funding(item: dict) -> Dict[str, Any]:
    delta = item.get("delta") if isinstance(item.get("delta"), dict) else {}
    if not delta and isinstance(item.get("ledgerUpdate"), dict):
        delta = item.get("ledgerUpdate") or {}
    amount = (
        safe_float(item.get("usdc"))
        or safe_float(item.get("amount"))
        or safe_float(delta.get("usdc"))
        or safe_float(delta.get("amount"))
        or safe_float(delta.get("delta"))
    )
    return {
        "time": safe_int(item.get("time")) or 0,
        "coin": str(item.get("coin") or delta.get("coin") or ""),
        "amount": amount,
        "type": str(item.get("type") or item.get("kind") or delta.get("type") or "funding"),
        "direction": "received" if amount is not None and amount >= 0 else "paid",
        "raw": item,
    }


def _parse_hyperliquid_position(item: dict) -> Dict[str, Any]:
    position = item.get("position") if isinstance(item.get("position"), dict) else item
    size = safe_float(position.get("szi"))
    if size is None:
        signed_size = safe_float(position.get("sz")) or safe_float(position.get("size"))
        size = abs(signed_size) if signed_size is not None else None
    raw_signed_size = safe_float(position.get("szi")) or safe_float(position.get("signedSz")) or safe_float(position.get("sz"))
    side = "flat"
    if raw_signed_size is not None:
        if raw_signed_size > 0:
            side = "long"
        elif raw_signed_size < 0:
            side = "short"
    entry_price = safe_float(position.get("entryPx"))
    mark_price = safe_float(position.get("markPx"))
    reference_price = mark_price if mark_price is not None else entry_price
    notional = abs(raw_signed_size or 0.0) * reference_price if raw_signed_size is not None and reference_price is not None else None
    leverage_value = None
    leverage = position.get("leverage")
    if isinstance(leverage, dict):
        leverage_value = safe_float(leverage.get("value"))
    else:
        leverage_value = safe_float(leverage)
    return {
        "coin": str(position.get("coin") or item.get("coin") or ""),
        "side": side,
        "size": abs(raw_signed_size) if raw_signed_size is not None else size,
        "signed_size": raw_signed_size,
        "entry_price": entry_price,
        "mark_price": mark_price,
        "liquidation_price": safe_float(position.get("liquidationPx")),
        "leverage": leverage_value,
        "max_leverage": safe_float(position.get("maxLeverage")),
        "margin_used": safe_float(position.get("marginUsed")),
        "position_value": safe_float(position.get("positionValue")) or notional,
        "unrealized_pnl": safe_float(position.get("unrealizedPnl")),
        "return_on_equity": safe_float(position.get("returnOnEquity")),
        "cum_funding": safe_float(position.get("cumFunding")),
        "raw": position,
    }


def fetch_hyperliquid_address_mode(
    address: str,
    coin: str,
    lookback_hours: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    normalized_address = normalize_onchain_address(address)
    if not is_valid_onchain_address(normalized_address):
        return {
            "status": "error",
            "error": "地址格式无效",
            "address": normalized_address,
            "positions": [],
            "fills": [],
            "funding": [],
            "active_asset": {},
            "spot_state": {},
        }

    client = HyperliquidClient(timeout=timeout)
    selected_coin = str(coin or "").strip().upper()
    now_ms = int(time.time() * 1000)
    start_time_ms = now_ms - max(1, int(lookback_hours)) * 3_600_000
    errors: List[str] = []

    try:
        state = client.fetch_clearinghouse_state(normalized_address)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "address": normalized_address,
            "positions": [],
            "fills": [],
            "funding": [],
            "active_asset": {},
            "spot_state": {},
        }

    raw_positions = state.get("assetPositions") or []
    positions = [_parse_hyperliquid_position(item) for item in raw_positions if isinstance(item, dict)]
    if selected_coin:
        positions = [item for item in positions if str(item.get("coin") or "").upper() == selected_coin]
    positions = sorted(positions, key=lambda item: abs(float(item.get("position_value") or 0.0)), reverse=True)

    try:
        raw_funding = client.fetch_user_funding(normalized_address, start_time_ms, now_ms)
    except Exception as exc:
        raw_funding = []
        errors.append(f"funding: {exc}")
    funding_records = [_parse_hyperliquid_funding(item) for item in raw_funding if isinstance(item, dict)]
    if selected_coin:
        funding_records = [item for item in funding_records if not item.get("coin") or str(item.get("coin")).upper() == selected_coin]
    funding_records = [item for item in funding_records if int(item.get("time") or 0) >= start_time_ms]
    funding_records.sort(key=lambda item: int(item.get("time") or 0), reverse=True)

    try:
        raw_fills = client.fetch_user_fills(normalized_address)
    except Exception as exc:
        raw_fills = []
        errors.append(f"fills: {exc}")
    fills = [_parse_hyperliquid_fill(item) for item in raw_fills if isinstance(item, dict)]
    if selected_coin:
        fills = [item for item in fills if str(item.get("coin") or "").upper() == selected_coin]
    fills = [item for item in fills if int(item.get("time") or 0) >= start_time_ms]
    fills.sort(key=lambda item: int(item.get("time") or 0), reverse=True)

    active_asset: Dict[str, Any] = {}
    if selected_coin:
        try:
            active_asset = client.fetch_active_asset_data(normalized_address, selected_coin)
        except Exception as exc:
            errors.append(f"activeAssetData: {exc}")

    spot_state: Dict[str, Any] = {}
    try:
        spot_state = client.fetch_spot_clearinghouse_state(normalized_address)
    except Exception as exc:
        errors.append(f"spotClearinghouseState: {exc}")

    role_payload: Dict[str, Any] = {}
    try:
        role_payload = client.fetch_user_role(normalized_address)
    except Exception as exc:
        errors.append(f"userRole: {exc}")

    portfolio: List[dict] = []
    try:
        portfolio = client.fetch_portfolio(normalized_address)
    except Exception as exc:
        errors.append(f"portfolio: {exc}")

    vault_equities: List[dict] = []
    try:
        vault_equities = client.fetch_user_vault_equities(normalized_address)
    except Exception as exc:
        errors.append(f"userVaultEquities: {exc}")

    vault_details: Dict[str, Any] = {}
    role_text = str(role_payload.get("role") or role_payload.get("type") or role_payload.get("userRole") or "")
    if role_text.lower() == "vault":
        try:
            vault_details = client.fetch_vault_details(normalized_address)
        except Exception as exc:
            errors.append(f"vaultDetails: {exc}")

    margin_summary = state.get("marginSummary") or {}
    cross_margin_summary = state.get("crossMarginSummary") or {}
    return {
        "status": "ok",
        "error": " | ".join(errors) if errors else None,
        "address": normalized_address,
        "coin": selected_coin,
        "timestamp_ms": safe_int(state.get("time")) or now_ms,
        "account_value": safe_float(margin_summary.get("accountValue")),
        "total_notional_position": safe_float(margin_summary.get("totalNtlPos")),
        "total_raw_usd": safe_float(margin_summary.get("totalRawUsd")),
        "total_margin_used": safe_float(margin_summary.get("totalMarginUsed")),
        "withdrawable": safe_float(state.get("withdrawable")),
        "cross_account_value": safe_float(cross_margin_summary.get("accountValue")),
        "cross_margin_used": safe_float(cross_margin_summary.get("totalMarginUsed")),
        "positions": positions,
        "fills": fills[:80],
        "funding": funding_records[:80],
        "active_asset": active_asset,
        "spot_state": spot_state,
        "role": role_text or None,
        "role_payload": role_payload,
        "portfolio": portfolio,
        "vault_equities": vault_equities,
        "vault_details": vault_details,
        "raw_state": state,
    }


def fetch_hyperliquid_predicted_fundings(timeout: int = DEFAULT_TIMEOUT) -> List[list]:
    client = HyperliquidClient(timeout=timeout)
    payload = client._request("POST", "/info", json={"type": "predictedFundings"})
    return payload if isinstance(payload, list) else []


def fetch_hyperliquid_all_mids(timeout: int = DEFAULT_TIMEOUT) -> Dict[str, str]:
    client = HyperliquidClient(timeout=timeout)
    payload = client._request("POST", "/info", json={"type": "allMids"})
    return payload if isinstance(payload, dict) else {}


def fetch_hyperliquid_spot_meta_and_asset_contexts(timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    client = HyperliquidClient(timeout=timeout)
    return client.fetch_spot_meta_and_asset_contexts()


def fetch_hyperliquid_perps_at_open_interest_cap(timeout: int = DEFAULT_TIMEOUT, dex: Optional[str] = None) -> Dict[str, Any]:
    client = HyperliquidClient(timeout=timeout)
    return client.fetch_perps_at_open_interest_cap(dex=dex)


def fetch_bybit_insurance_pool(coin: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    client = BybitClient(timeout=timeout)
    base_coin = _normalize_coin_code(coin)
    try:
        payload = client._request("GET", "/v5/market/insurance", params={"coin": base_coin} if base_coin else None)
    except Exception:
        payload = {}
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    items = result.get("list", []) if isinstance(result, dict) else []
    normalized_rows: List[Dict[str, Any]] = []
    total_value = 0.0
    total_balance = 0.0
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        value = safe_float(item.get("value")) or 0.0
        balance = safe_float(item.get("balance")) or 0.0
        total_value += value
        total_balance += balance
        normalized_rows.append(
            {
                "coin": str(item.get("coin") or base_coin),
                "symbols": str(item.get("symbols") or ""),
                "balance": balance,
                "value": value,
            }
        )
    return {
        "coin": base_coin,
        "updated_time_ms": safe_int(result.get("updatedTime")) if isinstance(result, dict) else None,
        "total_value": total_value if normalized_rows else None,
        "total_balance": total_balance if normalized_rows else None,
        "rows": normalized_rows,
    }


def build_clients(timeout: int = DEFAULT_TIMEOUT) -> Dict[str, BaseClient]:
    return {
        "bybit": BybitClient(timeout=timeout),
        "binance": BinanceClient(timeout=timeout),
        "okx": OkxClient(timeout=timeout),
        "hyperliquid": HyperliquidClient(timeout=timeout),
    }


def fetch_all_snapshots(symbol_map: Dict[str, str], timeout: int = DEFAULT_TIMEOUT) -> List[ExchangeSnapshot]:
    clients = build_clients(timeout=timeout)
    snapshots: List[ExchangeSnapshot] = []
    for key in EXCHANGE_ORDER:
        symbol = str(symbol_map.get(key) or "").strip().upper()
        if not symbol:
            snapshots.append(
                ExchangeSnapshot(
                    exchange=clients[key].exchange_name,
                    symbol="",
                    status="error",
                    error="未上架此币",
                )
            )
            continue
        snapshots.append(clients[key].fetch(symbol))
    return snapshots


def fetch_exchange_snapshot(exchange_key: str, symbol: str, timeout: int = DEFAULT_TIMEOUT) -> ExchangeSnapshot:
    clients = build_clients(timeout=timeout)
    if not str(symbol or "").strip():
        return ExchangeSnapshot(exchange=clients[exchange_key].exchange_name, symbol="", status="error", error="未上架此币")
    return clients[exchange_key].fetch(symbol)


def fetch_exchange_candles(exchange_key: str, symbol: str, interval: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[Candle]:
    if not str(symbol or "").strip():
        return []
    clients = build_clients(timeout=timeout)
    return clients[exchange_key].fetch_candles(symbol, interval, limit)


def fetch_exchange_orderbook(exchange_key: str, symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[OrderBookLevel]:
    if not str(symbol or "").strip():
        return []
    clients = build_clients(timeout=timeout)
    return clients[exchange_key].fetch_orderbook(symbol, limit)


def fetch_exchange_oi_history(exchange_key: str, symbol: str, interval: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[OIPoint]:
    if not str(symbol or "").strip():
        return []
    clients = build_clients(timeout=timeout)
    return clients[exchange_key].fetch_open_interest_history(symbol, interval, limit)


def fetch_exchange_liquidations(exchange_key: str, symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[LiquidationEvent]:
    if not str(symbol or "").strip():
        return []
    clients = build_clients(timeout=timeout)
    return clients[exchange_key].fetch_liquidations(symbol, limit)


def fetch_exchange_trades(exchange_key: str, symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[TradeEvent]:
    if not str(symbol or "").strip():
        return []
    clients = build_clients(timeout=timeout)
    return clients[exchange_key].fetch_trades(symbol, limit)


def fetch_binance_trader_sentiment(symbol: str, period: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, List[dict]]:
    client = BinanceClient(timeout=timeout)
    params = {"symbol": symbol, "period": period, "limit": max(10, min(limit, 500))}
    datasets: Dict[str, List[dict]] = {}
    endpoint_map = {
        "taker_ratio": "/futures/data/takerlongshortRatio",
        "top_position": "/futures/data/topLongShortPositionRatio",
        "top_account": "/futures/data/topLongShortAccountRatio",
        "global_account": "/futures/data/globalLongShortAccountRatio",
    }
    for dataset_key, path in endpoint_map.items():
        try:
            payload = client._request("GET", path, params=params)
            datasets[dataset_key] = payload if isinstance(payload, list) else []
        except Exception:
            datasets[dataset_key] = []
    return datasets


def fetch_bybit_trader_sentiment(symbol: str, period: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, List[dict]]:
    client = BybitClient(timeout=timeout)
    try:
        payload = client._request(
            "GET",
            "/v5/market/account-ratio",
            params={
                "category": "linear",
                "symbol": symbol,
                "period": BYBIT_RATIO_INTERVALS.get(period, "1h"),
                "limit": max(10, min(limit, 500)),
            },
        )
        items = payload.get("result", {}).get("list", [])
    except Exception:
        items = []

    normalized: List[dict] = []
    for item in reversed(items):
        long_share = safe_float(item.get("buyRatio"))
        short_share = safe_float(item.get("sellRatio"))
        ratio = None
        if long_share is not None and short_share not in (None, 0):
            ratio = long_share / short_share
        normalized.append(
            {
                "symbol": symbol,
                "timestamp": safe_int(item.get("timestamp")),
                "longShortRatio": ratio,
                "longAccount": long_share,
                "shortAccount": short_share,
                "buyRatio": long_share,
                "sellRatio": short_share,
            }
        )
    return {"account_ratio": normalized}


def fetch_binance_futures_orderbook_snapshot(symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> dict:
    client = BinanceClient(timeout=timeout)
    return client._request(
        "GET",
        "/fapi/v1/depth",
        params={"symbol": symbol, "limit": normalize_depth_limit("binance", max(5, min(limit, 1000)))},
    )


def fetch_binance_spot_orderbook_snapshot(symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> dict:
    client = BinanceSpotClient(timeout=timeout)
    return client._request(
        "GET",
        "/api/v3/depth",
        params={"symbol": symbol, "limit": normalize_depth_limit("binance", max(5, min(limit, 1000)))},
    )


def fetch_binance_spot_snapshot(symbol: str, timeout: int = DEFAULT_TIMEOUT) -> SpotSnapshot:
    client = BinanceSpotClient(timeout=timeout)
    return client.fetch(symbol)


def fetch_binance_spot_orderbook(symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[OrderBookLevel]:
    client = BinanceSpotClient(timeout=timeout)
    return client.fetch_orderbook(symbol, limit)


def fetch_binance_spot_trades(symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[TradeEvent]:
    client = BinanceSpotClient(timeout=timeout)
    return client.fetch_trades(symbol, limit)


def fetch_binance_basis_curve(pair: str, period: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, List[dict]]:
    client = BinanceClient(timeout=timeout)
    datasets: Dict[str, List[dict]] = {}
    for contract_type in ("PERPETUAL", "CURRENT_QUARTER", "NEXT_QUARTER"):
        try:
            payload = client._request(
                "GET",
                "/futures/data/basis",
                params={
                    "pair": pair,
                    "contractType": contract_type,
                    "period": period,
                    "limit": max(10, min(limit, 200)),
                },
            )
            datasets[contract_type] = payload if isinstance(payload, list) else []
        except Exception:
            datasets[contract_type] = []
    return datasets


def build_spot_clients(timeout: int = DEFAULT_TIMEOUT) -> Dict[str, BaseClient]:
    return {
        "binance": BinanceSpotClient(timeout=timeout),
        "bybit": BybitSpotClient(timeout=timeout),
        "okx": OkxSpotClient(timeout=timeout),
    }


def fetch_spot_snapshot(exchange_key: str, symbol: str, timeout: int = DEFAULT_TIMEOUT) -> SpotSnapshot:
    clients = build_spot_clients(timeout=timeout)
    if not str(symbol or "").strip():
        return SpotSnapshot(exchange=clients[exchange_key].exchange_name, symbol="", status="error", error="未上架此币")
    return clients[exchange_key].fetch(symbol)


def fetch_spot_orderbook(exchange_key: str, symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[OrderBookLevel]:
    if not str(symbol or "").strip():
        return []
    clients = build_spot_clients(timeout=timeout)
    return clients[exchange_key].fetch_orderbook(symbol, limit)


def fetch_spot_trades(exchange_key: str, symbol: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> List[TradeEvent]:
    if not str(symbol or "").strip():
        return []
    clients = build_spot_clients(timeout=timeout)
    client = clients[exchange_key]
    return client.fetch_trades(symbol, limit)  # type: ignore[attr-defined]


def snapshots_to_rows(snapshots: List[ExchangeSnapshot]):
    return [snapshot.to_row() for snapshot in snapshots]

