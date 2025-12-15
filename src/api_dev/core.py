"""Low-level safe async primitives for beamline control"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from bcs.BCSz import BCSServer, MotorStatus

from .types import MotorError, MotorTimeoutError, ScanAbortedError

# ============================================================================
# Abort Mechanism
# ============================================================================


class AbortFlag:
    """Thread-safe abort flag for scan cancellation in Jupyter notebooks"""

    def __init__(self):
        self._aborted = False
        self._lock = asyncio.Lock()

    async def set(self):
        """Set the abort flag"""
        async with self._lock:
            self._aborted = True

    async def is_set(self) -> bool:
        """Check if abort flag is set"""
        async with self._lock:
            return self._aborted

    async def clear(self):
        """Clear the abort flag"""
        async with self._lock:
            self._aborted = False


# ============================================================================
# Wait Utilities
# ============================================================================


async def wait_for_motors(
    server: BCSServer,
    motors: List[str],
    timeout: float = 30.0,
    check_interval: float = 0.05,
    abort_flag: Optional[AbortFlag] = None,
) -> None:
    """
    Wait for all motors to complete movement.

    Parameters:
        server: BCS server instance
        motors: List of motor names to wait for
        timeout: Maximum time to wait in seconds
        check_interval: Time between status checks
        abort_flag: Optional abort flag to check

    Raises:
        MotorTimeoutError: If timeout exceeded
        ScanAbortedError: If abort_flag is set
    """
    start_time = time.time()

    while True:
        # Check for abort
        if abort_flag and await abort_flag.is_set():
            raise ScanAbortedError("Scan aborted by user")

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise MotorTimeoutError(
                f"Motors {motors} did not complete within {timeout}s"
            )

        # Check motor status
        response = await server.get_motor(motors=motors)
        all_complete = True

        for motor_data in response["data"]:
            status = MotorStatus(motor_data["status"])
            if not status.is_set(MotorStatus.MOVE_COMPLETE):
                all_complete = False
                break

        if all_complete:
            return

        await asyncio.sleep(check_interval)


async def wait_for_settle(delay: float, abort_flag: Optional[AbortFlag] = None) -> None:
    """
    Wait for motor settling with abort check.

    Parameters:
        delay: Time to wait in seconds
        abort_flag: Optional abort flag to check

    Raises:
        ScanAbortedError: If abort_flag is set during wait
    """
    if delay <= 0:
        return

    # Check abort every 100ms
    steps = int(delay / 0.1)
    for _ in range(steps):
        if abort_flag and await abort_flag.is_set():
            raise ScanAbortedError("Scan aborted during settle")
        await asyncio.sleep(0.1)

    # Sleep remainder
    remainder = delay - (steps * 0.1)
    if remainder > 0:
        if abort_flag and await abort_flag.is_set():
            raise ScanAbortedError("Scan aborted during settle")
        await asyncio.sleep(remainder)


# ============================================================================
# Context Managers
# ============================================================================


@asynccontextmanager
async def motor_move(
    server: BCSServer,
    motors: Dict[str, float],
    timeout: float = 30.0,
    backlash: bool = True,
    restore_on_exit: bool = True,
):
    """
    Context manager for safe motor movements with automatic position restoration.

    Parameters:
        server: BCS server instance
        motors: Dictionary of motor_name -> target_position
        timeout: Timeout for motor moves
        backlash: Use backlash compensation
        restore_on_exit: Restore initial positions on exit

    Yields:
        Dictionary of initial motor positions

    Raises:
        MotorError: If motor move fails

    Example:
        async with motor_move(server, {"Sample X": 10.0}) as initial_pos:
            # Do work at new position
            pass
        # Motors automatically return to initial_pos
    """
    # Record initial positions
    initial_response = await server.get_motor(motors=list(motors.keys()))
    initial_pos = {m["motor"]: m["position"] for m in initial_response["data"]}

    try:
        # Move to target positions
        command = "Backlash Move"  # if backlash else "Backlash Move"
        await server.command_motor(
            commands=[command] * len(motors),
            motors=list(motors.keys()),
            goals=list(motors.values()),
        )

        # Wait for completion
        await wait_for_motors(server, list(motors.keys()), timeout=timeout)

        yield initial_pos

    except Exception as e:
        raise MotorError(f"Motor move failed: {e}") from e

    finally:
        # Return to initial positions if requested
        if restore_on_exit:
            try:
                await server.command_motor(
                    commands=["Backlash Move"] * len(initial_pos),
                    motors=list(initial_pos.keys()),
                    goals=list(initial_pos.values()),
                )
                await wait_for_motors(server, list(initial_pos.keys()), timeout=timeout)
            except Exception as e:
                # Log but don't raise - we're in cleanup
                print(f"Warning: Failed to restore motor positions: {e}")


@asynccontextmanager
async def shutter_control(
    server: BCSServer, shutter: str = "Light Output", delay_before_open: float = 0.0
):
    """
    Context manager for safe shutter control.
    Guarantees shutter closes even on exception.

    Parameters:
        server: BCS server instance
        shutter: Shutter DIO channel name
        delay_before_open: Delay before opening shutter

    Example:
        async with shutter_control(server):
            # Shutter is open, collect data
            await server.acquire_data(time=1.0)
        # Shutter automatically closes
    """
    try:
        # Wait for settling
        if delay_before_open > 0:
            await asyncio.sleep(delay_before_open)

        # Open shutter
        await server.set_do(chan=shutter, value=True)
        yield

    finally:
        # Always close shutter
        try:
            await server.set_do(chan=shutter, value=False)
        except Exception as e:
            print(f"Warning: Failed to close shutter: {e}")
