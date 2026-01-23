"""Connection management for RsoxsServer in MCP context."""

import asyncio
from typing import Optional

try:
    from api_dev.server import RsoxsServer
except ImportError:
    from ..api_dev.server import RsoxsServer


class ConnectionManager:
    """Singleton connection manager for RsoxsServer."""

    _instance: Optional["ConnectionManager"] = None
    _lock = asyncio.Lock()
    _server: Optional[RsoxsServer] = None
    _connected: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_server(self) -> RsoxsServer:
        """
        Get or create RsoxsServer instance.

        Returns
        -------
        RsoxsServer
            Connected server instance

        Raises
        ------
        ConnectionError
            If connection fails
        """
        async with self._lock:
            if self._server is None or not self._connected:
                self._server = await RsoxsServer.create()
                self._connected = True
            return self._server

    async def ensure_connected(self) -> None:
        """
        Ensure server is connected, reconnect if needed.

        Raises
        ------
        ConnectionError
            If reconnection fails
        RuntimeError
            If connection state is inconsistent
        """
        async with self._lock:
            if self._server is None or not self._connected:
                try:
                    self._server = await RsoxsServer.create()
                    self._connected = True
                except ConnectionError as e:
                    self._connected = False
                    self._server = None
                    raise ConnectionError(f"Failed to connect to beamline server: {e}") from e
                except Exception as e:
                    self._connected = False
                    self._server = None
                    raise RuntimeError(f"Unexpected error connecting to beamline server: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from server."""
        async with self._lock:
            self._connected = False
            self._server = None

    @property
    def is_connected(self) -> bool:
        """Check if server is connected."""
        return self._connected and self._server is not None


connection_manager = ConnectionManager()
