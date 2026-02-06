"""Websocket connection manager with reconnection logic."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, List, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from packages.common.ws_status import (
    set_ws_connected,
    set_ws_reconnect_count,
    set_ws_connection_count,
    set_ws_streams,
    reset_ws_status,
    refresh_ws_heartbeat,
)

logger = logging.getLogger(__name__)

# Connection settings optimized for Binance Futures
MAX_STREAMS_PER_CONNECTION = 25  # Split across multiple connections for stability
MAX_RECONNECT_DELAY_SEC = 60
PING_INTERVAL_SEC = 20  # 20 seconds - Binance recommends frequent pings
PING_TIMEOUT_SEC = 30  # 30 seconds timeout
CLOSE_TIMEOUT_SEC = 10  # Close timeout
HEARTBEAT_INTERVAL_SEC = 30  # Heartbeat to keep status alive


class BinanceWebsocketConnection:
    """Manages a single websocket connection to Binance with reconnection."""

    def __init__(
        self,
        base_url: str,
        streams: List[str],
        on_message: Callable[[dict], None],
        connection_id: int = 0,
        manager: Optional["WebsocketManager"] = None,
    ):
        """
        Initialize websocket connection.

        Args:
            base_url: Websocket base URL (e.g., wss://fstream.binance.com)
            streams: List of stream names to subscribe
            on_message: Callback for incoming messages
            connection_id: Unique ID for this connection
            manager: Parent manager for connection counting
        """
        if len(streams) > MAX_STREAMS_PER_CONNECTION:
            raise ValueError(
                f"Too many streams ({len(streams)}). "
                f"Max allowed: {MAX_STREAMS_PER_CONNECTION}"
            )

        self.base_url = base_url
        self.streams = streams
        self.on_message = on_message
        self.connection_id = connection_id
        self.manager = manager
        self.ws: WebSocketClientProtocol | None = None
        self.should_run = True
        self.reconnect_count = 0
        self.is_connected = False

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
                    f"[Conn {self.connection_id}] Connecting to Binance websocket... "
                    f"(streams: {len(self.streams)}, attempt: {self.reconnect_count + 1})"
                )

                async with websockets.connect(
                    self.url,
                    ping_interval=PING_INTERVAL_SEC,
                    ping_timeout=PING_TIMEOUT_SEC,
                    close_timeout=CLOSE_TIMEOUT_SEC,
                ) as ws:
                    self.ws = ws
                    self.reconnect_count = 0
                    retry_delay = 1
                    self.is_connected = True

                    logger.info(f"[Conn {self.connection_id}] ✓ Websocket connected successfully")

                    # Notify manager of connection
                    if self.manager:
                        self.manager.on_connection_change()

                    await self._handle_messages()

            except websockets.ConnectionClosed as e:
                logger.warning(f"[Conn {self.connection_id}] Connection closed: {e}")
                self._handle_disconnect()

            except Exception as e:
                logger.error(f"[Conn {self.connection_id}] Connection error: {e}")
                self._handle_disconnect()

            if self.should_run:
                retry_delay = min(2 ** self.reconnect_count, MAX_RECONNECT_DELAY_SEC)
                logger.info(f"[Conn {self.connection_id}] Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)

    def _handle_disconnect(self) -> None:
        """Handle disconnection."""
        self.reconnect_count += 1
        self.is_connected = False
        if self.manager:
            self.manager.on_connection_change()

    async def _handle_messages(self) -> None:
        """Process incoming websocket messages."""
        async for message in self.ws:
            try:
                data = json.loads(message)
                self.on_message(data)
            except json.JSONDecodeError as e:
                logger.error(f"[Conn {self.connection_id}] Failed to parse message: {e}")
            except Exception as e:
                logger.error(f"[Conn {self.connection_id}] Error handling message: {e}")

    def stop(self) -> None:
        """Stop the connection gracefully."""
        self.should_run = False
        self.is_connected = False
        if self.ws:
            asyncio.create_task(self.ws.close())


class WebsocketManager:
    """Manages multiple websocket connections with status tracking."""

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
        self._heartbeat_task: asyncio.Task | None = None
        self._status_lock = asyncio.Lock()

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
                connection_id=len(self.connections),
                manager=self,
            )
            self.connections.append(conn)

        logger.info(
            f"Created {len(self.connections)} connection(s) for {len(streams)} stream(s)"
        )

    def on_connection_change(self) -> None:
        """Called when any connection's status changes."""
        connected_count = sum(1 for c in self.connections if c.is_connected)
        total_count = len(self.connections)

        # Update connection count
        set_ws_connection_count(connected_count)

        # Calculate stream counts
        active_streams = sum(len(c.streams) for c in self.connections if c.is_connected)
        total_streams = sum(len(c.streams) for c in self.connections)
        set_ws_streams(active_streams, total_streams)

        # Update overall status
        if connected_count == total_count:
            # All connections are up
            set_ws_connected(True)
            set_ws_reconnect_count(0)
        elif connected_count > 0:
            # Some connections are up
            set_ws_connected(True)
            total_reconnects = sum(c.reconnect_count for c in self.connections)
            set_ws_reconnect_count(total_reconnects)
        else:
            # No connections
            reset_ws_status()
            total_reconnects = sum(c.reconnect_count for c in self.connections)
            set_ws_reconnect_count(total_reconnects)

        logger.info(f"Connection status: {connected_count}/{total_count} connected")

    async def _heartbeat_loop(self) -> None:
        """Periodically refresh heartbeat to keep status alive in Redis."""
        logger.info(f"Starting heartbeat loop (interval: {HEARTBEAT_INTERVAL_SEC}s)")
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)

                # Refresh heartbeat if at least one connection is up
                connected_count = sum(1 for c in self.connections if c.is_connected)
                if connected_count > 0:
                    refresh_ws_heartbeat()

            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def start(self) -> None:
        """Start all connections and heartbeat task."""
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start all connections
        tasks = [conn.connect() for conn in self.connections]
        await asyncio.gather(*tasks)

    def stop(self) -> None:
        """Stop all connections and heartbeat task."""
        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Stop all connections
        for conn in self.connections:
            conn.stop()
