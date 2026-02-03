"""Test websocket connection to Binance Futures."""
import asyncio
import logging

from packages.common.config import get_settings
from services.websocket.connection import WebsocketManager
from services.websocket.stream_router import StreamRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TestHandler:
    """Test handler to count messages."""

    def __init__(self):
        self.counts = {}
        self.total = 0

    def handle(self, data: dict) -> None:
        """Handle any message."""
        event_type = data.get("e", "unknown")
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        self.total += 1

        if self.total % 100 == 0:
            logger.info(f"Received {self.total} messages. Counts: {self.counts}")


async def test_connection():
    """Test websocket connection with a few symbols."""
    settings = get_settings()

    # Test with 2 symbols only
    test_symbols = ["BTCUSDT", "ETHUSDT"]

    # Build streams
    streams = []
    for symbol in test_symbols:
        s = symbol.lower()
        streams.extend([
            f"{s}@kline_1m",
            f"{s}@kline_15m",
            f"{s}@kline_1h",
            f"{s}@markPrice@1s",
            f"{s}@bookTicker",
        ])

    logger.info(f"Testing with {len(streams)} streams for {len(test_symbols)} symbols")
    logger.info(f"URL: {settings.binance_futures_ws_url}")

    # Setup router and handler
    router = StreamRouter()
    test_handler = TestHandler()

    router.register("kline_1m", test_handler.handle)
    router.register("kline_15m", test_handler.handle)
    router.register("kline_1h", test_handler.handle)
    router.register("markPrice@1s", test_handler.handle)
    router.register("bookTicker", test_handler.handle)

    # Setup websocket manager
    ws_manager = WebsocketManager(settings.binance_futures_ws_url, router.route)
    ws_manager.add_streams(streams)

    logger.info("Starting websocket connection...")
    logger.info("Press Ctrl+C to stop")

    try:
        await ws_manager.start()
    except KeyboardInterrupt:
        logger.info("Stopping...")
        ws_manager.stop()
        await asyncio.sleep(1)

    logger.info(f"Final stats: {test_handler.counts}")
    logger.info(f"Total messages: {test_handler.total}")
    logger.info(f"Router stats: {router.get_stats()}")


if __name__ == "__main__":
    asyncio.run(test_connection())
