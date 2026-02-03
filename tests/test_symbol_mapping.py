import pytest

from packages.common.symbol_map import to_internal, to_ws, to_rest, validate_internal


def test_symbol_mapping():
    assert to_internal("btcusdt", "ws") == "BTCUSDT"
    assert to_internal("BTCUSDT", "rest") == "BTCUSDT"
    assert to_internal("BTC/USDT:USDT", "ccxt") == "BTCUSDT"
    assert to_ws("BTCUSDT") == "btcusdt"
    assert to_rest("BTCUSDT") == "BTCUSDT"


def test_invalid_internal():
    with pytest.raises(ValueError):
        validate_internal("btc/usdt")
