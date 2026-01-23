"""MCP tools for analog input (AI) channels."""

from typing import Any

from ..connection import connection_manager
from ..models import AIChannelResponse, AIUncertaintyResponse, AIValuesResponse


async def list_ai_channels() -> dict[str, Any]:
    """
    List all available AI channel names.

    Returns
    -------
    dict
        Response with channels list
    """
    try:
        await connection_manager.ensure_connected()
        server = await connection_manager.get_server()

        result = await server.list_ais()
        if not result.get("success", False):
            raise RuntimeError(f"Failed to list AI channels: {result.get('error description', 'Unknown error')}")

        channels = result.get("chans", [])
        response = AIChannelResponse(channels=channels)
        return response.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error listing AI channels: {e}") from e


async def get_ai_values(channels: list[str] | None = None) -> dict[str, Any]:
    """
    Get current values for AI channels.

    Parameters
    ----------
    channels : list[str] | None, optional
        List of channel names to get values for. If None or empty, gets all channels.

    Returns
    -------
    dict
        Response with channel values

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
            list_result = await server.list_ais()
            if not list_result.get("success", False):
                error_msg = list_result.get("error description", "Unknown error")
                raise RuntimeError(f"Failed to list AI channels: {error_msg}")
            channels = list_result.get("chans", [])

        if not channels:
            return AIValuesResponse(values={}).model_dump()

        try:
            result = await server.ai.table(channels)
        except KeyError as e:
            raise ValueError(f"Invalid AI channel name: {e}") from e

        values = {}

        for chan_data in result.status.to_dict("records"):
            chan_name = chan_data.get("chan", "")
            if not chan_name:
                continue
            data_array = chan_data.get("data", [])
            if isinstance(data_array, list) and len(data_array) > 0:
                try:
                    values[chan_name] = float(data_array[-1])
                except (ValueError, TypeError):
                    values[chan_name] = 0.0
            elif isinstance(data_array, (int, float)):
                values[chan_name] = float(data_array)
            else:
                values[chan_name] = 0.0

        response = AIValuesResponse(values=values)
        return response.model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting AI values: {e}") from e


async def get_ai_with_uncertainty(
    channels: list[str],
    acquisition_time: float = 1.0,
) -> dict[str, Any]:
    """
    Get AI channel values with uncertainty (mean and standard deviation).

    Parameters
    ----------
    channels : list[str]
        List of channel names to acquire
    acquisition_time : float, optional
        Acquisition time in seconds (default: 1.0)

    Returns
    -------
    dict
        Response with channel values including mean and std
    """
    try:
        await connection_manager.ensure_connected()
        server = await connection_manager.get_server()

        if not channels:
            raise ValueError("channels list cannot be empty")

        if acquisition_time <= 0:
            raise ValueError("acquisition_time must be positive")

        ufloat_data = await server.ai.get_with_uncertainty(keys=channels, acquisition_time=acquisition_time)

        values = {}
        for chan, uval in ufloat_data.items():
            values[chan] = {
                "mean": float(uval.nominal_value),
                "std": float(uval.std_dev),
            }

        response = AIUncertaintyResponse(values=values)
        return response.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error getting AI values with uncertainty: {e}") from e
