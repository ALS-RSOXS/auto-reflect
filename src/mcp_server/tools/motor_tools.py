"""MCP tools for motor positions and status."""

from typing import Any

from ..connection import connection_manager
from ..models import MotorListResponse, MotorPositionsResponse, MotorStatusResponse


async def list_motors() -> dict[str, Any]:
    """
    List all available motor names.

    Returns
    -------
    dict
        Response with motors list
    """
    try:
        await connection_manager.ensure_connected()
        server = await connection_manager.get_server()

        result = await server.list_motors()
        if not result.get("success", False):
            raise RuntimeError(f"Failed to list motors: {result.get('error description', 'Unknown error')}")

        motors = result.get("motors", [])
        response = MotorListResponse(motors=motors)
        return response.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error listing motors: {e}") from e


async def get_motor_positions(motors: list[str] | None = None) -> dict[str, Any]:
    """
    Get current positions for motors.

    Parameters
    ----------
    motors : list[str] | None, optional
        List of motor names to get positions for. If None or empty, gets all motors.

    Returns
    -------
    dict
        Response with motor positions

    Raises
    ------
    RuntimeError
        If server communication fails
    ValueError
        If invalid motor names are provided
    """
    try:
        await connection_manager.ensure_connected()
        server = await connection_manager.get_server()

        if not motors:
            list_result = await server.list_motors()
            if not list_result.get("success", False):
                error_msg = list_result.get("error description", "Unknown error")
                raise RuntimeError(f"Failed to list motors: {error_msg}")
            motors = list_result.get("motors", [])

        if not motors:
            return MotorPositionsResponse(positions={}).model_dump()

        try:
            result = await server.motor.table(motors)
        except KeyError as e:
            raise ValueError(f"Invalid motor name: {e}") from e

        positions = {}

        for motor_data in result.status.to_dict("records"):
            motor_name = motor_data.get("motor", "")
            if not motor_name:
                continue
            try:
                position = motor_data.get("position", 0.0)
                positions[motor_name] = float(position)
            except (ValueError, TypeError):
                positions[motor_name] = 0.0

        response = MotorPositionsResponse(positions=positions)
        return response.model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting motor positions: {e}") from e


async def get_motor_status(motors: list[str] | None = None) -> dict[str, Any]:
    """
    Get full status for motors including position, goal, and status bits.

    Parameters
    ----------
    motors : list[str] | None, optional
        List of motor names to get status for. If None or empty, gets all motors.

    Returns
    -------
    dict
        Response with motor status information

    Raises
    ------
    RuntimeError
        If server communication fails
    ValueError
        If invalid motor names are provided
    """
    try:
        await connection_manager.ensure_connected()
        server = await connection_manager.get_server()

        if not motors:
            list_result = await server.list_motors()
            if not list_result.get("success", False):
                error_msg = list_result.get("error description", "Unknown error")
                raise RuntimeError(f"Failed to list motors: {error_msg}")
            motors = list_result.get("motors", [])

        if not motors:
            return MotorStatusResponse(status={}).model_dump()

        try:
            result = await server.motor.table(motors)
        except KeyError as e:
            raise ValueError(f"Invalid motor name: {e}") from e

        status_dict = {}

        for motor_data in result.status.to_dict("records"):
            motor_name = motor_data.get("motor", "")
            if not motor_name:
                continue
            try:
                status_dict[motor_name] = {
                    "position": float(motor_data.get("position", 0.0)),
                    "goal": float(motor_data.get("goal", 0.0)),
                    "status": int(motor_data.get("status", 0)),
                }
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid motor data for {motor_name}: {e}") from e

        response = MotorStatusResponse(status=status_dict)
        return response.model_dump()

    except (ConnectionError, RuntimeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error getting motor status: {e}") from e
