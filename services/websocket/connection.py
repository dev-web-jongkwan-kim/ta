"""Websocket connection manager with reconnection logic."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, List

import websockets
from websockets.client import WebSocketClientProtocol

from packages.common.ws_status import set_ws_connected, set_ws_reconnect_count, reset_ws_status

logger = logging.getLogger(__name__)

MAX_STREAMS_PER_CONNECTION = 200  # Conservative limit (Binance allows 1024)
MAX_RECONNECT_DELAY_SEC = 60
PING_INTERVAL_SEC = 180  # 3 minutes
PING_TIMEOUT_SEC = 600  # 10 minutes


class BinanceWebsocketConnection:
    """Manages a single websocket connection to Binance with reconnection."""

    def __init__(
        self,
        base_url: str,
        streams: List[str],
        on_message: Callable[[dict], None],
    ):
        """
        Initialize websocket connection.

        Args:
            base_url: Websocket base URL (e.g., wss://fstream.binance.com)
            streams: List of stream names to subscribe
            on_message: Callback for incoming messages
        """
        if len(streams) > MAX_STREAMS_PER_CONNECTION:
            raise ValueError(
                f"Too many streams ({len(streams)}). "
                f"Max allowed: {MAX_STREAMS_PER_CONNECTION}"
            )

        self.base_url = base_url
        self.streams = streams
        self.on_message = on_message
        self.ws: WebSocketClientProtocol | None = None
        self.should_run = True
        self.reconnect_count = 0

    @property
    def url(self) -> str:
        """Build combined stream URL."""
        stream_str = "/".join(self.streams)
        return f"{self.base_url}/stream?streams={stream_str}"

    async def connect(self) -> None:
        """Connect with exponential backoff retry."""
        retry_delay = 1

        while self.should_run:
            try:
                logger.info(
                    f"Connecting to Binance websocket... "
                    f"(streams: {len(self.streams)}, attempt: {self.reconnect_count + 1})"
                )

                async with websockets.connect(
                    self.url,
                    ping_interval=PING_INTERVAL_SEC,
                    ping_timeout=PING_TIMEOUT_SEC,
                ) as ws:
                    self.ws = ws
                    self.reconnect_count = 0
                    retry_delay = 1

                    logger.info("✓ Websocket connected successfully")
                    set_ws_connected(True)
                    set_ws_reconnect_count(0)
                    await self._handle_messages()

            except websockets.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                self.reconnect_count += 1
                reset_ws_status()
                set_ws_reconnect_count(self.reconnect_count)

            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.reconnect_count += 1
                reset_ws_status()
                set_ws_reconnect_count(self.reconnect_count)

            if self.should_run:
                retry_delay = min(2 ** self.reconnect_count, MAX_RECONNECT_DELAY_SEC)
                logger.info(f"Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)

    async def _handle_messages(self) -> None:
        """Process incoming websocket messages."""
        async for message in self.ws:
            try:
                data = json.loads(message)
                self.on_message(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")

    def stop(self) -> None:
        """Stop the connection gracefully."""
        self.should_run = False
        if self.ws:
            asyncio.create_task(self.ws.close())


class WebsocketManager:
    """Manages multiple websocket connections."""

    def __init__(self, base_url: str, on_message: Callable[[dict], None]):
        """
        Initialize websocket manager.

        Args:
            base_url: Websocket base URL
            on_message: Callback for all incoming messages
        """
        self.base_url = base_url
        self.on_message = on_message
        self.connections: List[BinanceWebsocketConnection] = []

    def add_streams(self, streams: List[str]) -> None:
        """
        Add streams to the manager.

        Automatically creates multiple connections if needed.
        """
        # Split streams into chunks of MAX_STREAMS_PER_CONNECTION
        for i in range(0, len(streams), MAX_STREAMS_PER_CONNECTION):
            chunk = streams[i : i + MAX_STREAMS_PER_CONNECTION]
            conn = BinanceWebsocketConnection(
                self.base_url,
                chunk,
                self.on_message,
            )
            self.connections.append(conn)

        logger.info(
            f"Created {len(self.connections)} connection(s) for {len(streams)} stream(s)"
        )

    async def start(self) -> None:
        """Start all connections."""
        tasks = [conn.connect() for conn in self.connections]
        await asyncio.gather(*tasks)

    def stop(self) -> None:
        """Stop all connections."""
        for conn in self.connections:
            conn.stop()
