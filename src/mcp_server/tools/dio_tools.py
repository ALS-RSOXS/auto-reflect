"""MCP tools for digital I/O (DIO) channels."""

from typing import Any

from ..connection import connection_manager
from ..models import DIOChannelsResponse, DIOStatesResponse


async def list_dio_channels() -> dict[str, Any]:
    """
    List all available DIO channel names.

    Returns
    -------
    dict
        Response with channels list
    """
    try:
        await connection_manager.ensure_connected()
        server = await connection_manager.get_server()

        result = await server.list_dios()
        if not result.get("success", False):
            raise RuntimeError(f"Failed to list DIO channels: {result.get('error description', 'Unknown error')}")

        channels = result.get("chans", [])
        response = DIOChannelsResponse(channels=channels)
        return response.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error listing DIO channels: {e}") from e


async def get_dio_states(channels: list[str] | None = None) -> dict[str, Any]:
    """
    Get current states for DIO channels.

    Parameters
    ----------
    channels : list[str] | None, optional
        List of channel names to get states for. If None or empty, gets all channels.

    Returns
    -------
    dict
        Response with channel states (boolean values)

    Raises
    ------
    RuntimeError
        If server communication fails
    ValueError
        If invalid channel names are provided
    """
    try:
        await connection_manager.ensure_connected()
        server = await connection_manager.get_server()

        if not channels:
            list_result = await server.list_dios()
            if not list_result.get("success", False):
                error_msg = list_result.get("error description", "Unknown error")
                raise RuntimeError(f"Failed to list DIO channels: {error_msg}")
            channels = list_result.get("chans", [])

        if not channels:
            return DIOStatesResponse(states={}).model_dump()

        try:
            result = await server.dio.table(channels)
        except KeyError as e:
            raise ValueError(f"Invalid DIO channel name: {e}") from e

        states = {}

        for dio_data in result.status.to_dict("records"):
            chan_name = dio_data.get("chan", "")
            if not chan_name:
                continue
            data_value = dio_data.get("data", False)
            states[chan_name] = bool(data_value)

        response = DIOStatesResponse(states=states)
        return response.model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting DIO states: {e}") from e
