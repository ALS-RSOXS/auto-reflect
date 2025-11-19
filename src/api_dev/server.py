"""
High-level RSoXS beamline server interface.

This module provides the RsoxsServer class which extends BCSz.BCSServer
with high-level scan functionality and type-safe accessors.
"""

import asyncio
import io
from contextlib import redirect_stdout

import pandas as pd
from bcs import BCSz
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .accessors import RsoxsAccessor
from .scan import ScanExecutor, ScanPlan
from .types import AI, DIO, Motor


class Connection(BaseSettings):
    """Configuration for BCS server connection from environment variables."""

    addr: str = Field(alias="BCS_SERVER_ADDRESS")
    port: int = Field(alias="BCS_SERVER_PORT")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RsoxsServer(BCSz.BCSServer):
    """
    High-level interface for RSoXS beamline control.

    Provides type-safe accessors for AI channels, motors, and digital I/O,
    plus high-level scan orchestration with DataFrame interface and
    automatic uncertainty propagation.

    Attributes
    ----------
    ai : RsoxsAccessor[AI]
        Read-only accessor for analog input channels
    motor : RsoxsAccessor[Motor]
        Accessor for motor positions
    dio : RsoxsAccessor[DIO]
        Accessor for digital I/O channels
    ccd_ready : bool
        Flag indicating if CCD camera is ready

    Examples
    --------
    >>> # Connect to beamline
    >>> server = await RsoxsServer.create()
    >>>
    >>> # Simple AI acquisition with uncertainty
    >>> data = await server.ai.get_with_uncertainty(
    ...     keys=["Photodiode", "TEY signal"],
    ...     acquisition_time=1.0
    ... )
    >>>
    >>> # DataFrame-based scan
    >>> scan_df = pd.DataFrame({
    ...     "Sample X": [10.0, 10.5, 11.0],
    ...     "exposure": [1.0, 1.0, 1.5]
    ... })
    >>> results = await server.scan_from_dataframe(
    ...     df=scan_df,
    ...     ai_channels=["Photodiode"]
    ... )
    """

    CONFIG = Connection()

    def __init__(self):
        super().__init__()

        self.ccd_ready: bool = False

        # Type-safe accessors
        self.ai = RsoxsAccessor[AI](self, AI, readonly=True)
        self.motor = RsoxsAccessor[Motor](self, Motor)
        self.dio = RsoxsAccessor[DIO](self, DIO)

        # Scan executor
        self._scan_executor = ScanExecutor(self)

    async def connect_with_env(self):
        """
        Connect to the server using environment variables.

        Reads BCS_SERVER_ADDRESS and BCS_SERVER_PORT from .env file
        and establishes connection to the beamline control system.

        Raises
        ------
        ConnectionError
            If connection fails
        """
        buff = io.StringIO()
        with redirect_stdout(buff):
            await self.connect(**self.CONFIG.model_dump())
            self.__public_key = buff.getvalue().split(" ")[-1]

    @classmethod
    async def create(cls) -> "RsoxsServer":
        """
        Factory method to create and connect the server.

        Returns
        -------
        RsoxsServer
            Connected server instance

        Examples
        --------
        >>> server = await RsoxsServer.create()
        """
        instance = cls()
        await instance.connect_with_env()
        return instance

    # -------- Instrument Setup Methods --------

    async def setup_ccd(self) -> None:
        """
        Setup the CCD camera for data acquisition.

        Checks if CCD driver is running and updates ccd_ready flag.

        Raises
        ------
        RuntimeError
            If CCD status check fails
        """
        if self.ccd_ready:
            return

        res = await self.get_instrument_driver_status(name="CCD")
        if not res["success"]:
            raise RuntimeError(f"Failed to get CCD status: {res['error description']}")
        self.ccd_ready = res["running"]

    async def _set_ccd_temp(self, target_temp: float = -40.0) -> None:
        """
        Set and wait for CCD temperature to stabilize.

        Parameters
        ----------
        target_temp : float, optional
            Target temperature in Celsius (default: -40.0)

        Raises
        ------
        RuntimeError
            If CCD is not setup
        TimeoutError
            If temperature does not stabilize in time
        """
        if not self.ccd_ready:
            raise RuntimeError("CCD is not setup. Call setup_ccd() first.")

        current_temp = (await self.get_state_variable("Camera Temp"))["numeric_value"]
        set_temp = (await self.get_state_variable("Camera Temp Setpoint"))[
            "numeric_value"
        ]

        if set_temp > target_temp:
            print(f"Camera temperature setpoint too high, setting to {target_temp}C")
            set_temp = target_temp
            await self.set_state_variable("Camera Temp Setpoint", target_temp)

        # Wait for camera to reach target temperature
        tolerance = 1.0  # Temperature tolerance in degrees C
        max_wait_time = 300  # Maximum wait time in seconds
        check_interval = 2  # Check every 2 seconds

        elapsed_time = 0
        while abs(current_temp - set_temp) > tolerance:
            if elapsed_time >= max_wait_time:
                raise TimeoutError(
                    f"Camera temperature did not stabilize within {max_wait_time}s. "
                    f"Current: {current_temp}C, Target: {set_temp}C"
                )

            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
            current_temp = (await self.get_state_variable("Camera Temp"))[
                "numeric_value"
            ]
            print(
                f"Waiting for camera to cool... "
                f"Current: {current_temp:.1f}C, Target: {set_temp}C"
            )

        print(f"Camera temperature stabilized at {current_temp:.1f}C")

    # -------- High-Level Scan Methods --------

    async def scan_from_dataframe(
        self,
        df: pd.DataFrame,
        ai_channels: list[AI] | None = None,
        default_delay: float = 0.1,
        shutter: DIO = "Shutter Output",
        motor_timeout: float = 30.0,
        progress: bool = True,
        actuate_every: bool = True,
    ) -> pd.DataFrame:
        """
        Execute a scan defined by a DataFrame with automatic uncertainty.

        The DataFrame should have:
        - Motor columns: Column names matching Motor type literal
        - Exposure column: "exposure", "exp", "count_time", or similar
        - Each row defines one scan point

        Parameters
        ----------
        df : pd.DataFrame
            Scan definition with motor positions and exposure times
        ai_channels : list[AI] | None, optional
            AI channels to acquire at each point (default: None)
        default_delay : float, optional
            Delay after motor move in seconds (default: 0.2)
        shutter : DIO, optional
            Shutter DIO channel name (default: "Light Output")
        motor_timeout : float, optional
            Timeout for motor moves in seconds (default: 30.0)
        progress : bool, optional
            Show progress bar (default: True)

        Returns
        -------
        pd.DataFrame
            Results with motor positions and AI data (mean and std columns)

        Examples
        --------
        >>> scan_df = pd.DataFrame({
        ...     "Sample X": [10.0, 10.5, 11.0],
        ...     "Sample Y": [0.0, 0.0, 0.0],
        ...     "exposure": [1.0, 1.5, 2.0]
        ... })
        >>> results = await server.scan_from_dataframe(
        ...     df=scan_df,
        ...     ai_channels=["Photodiode", "TEY signal"]
        ... )
        >>> print(results.columns)
        Index(['index', 'Sample X_position', 'Sample Y_position',
               'Photodiode_mean', 'Photodiode_std',
               'TEY signal_mean', 'TEY signal_std',
               'exposure', 'timestamp'])
        """
        # Create scan plan
        scan_plan = ScanPlan.from_dataframe(
            df=df,
            ai_channels=ai_channels or [],
            default_delay=default_delay,
            shutter=shutter,
        )

        # Execute scan
        results = await self._scan_executor.execute_scan(scan_plan, progress=progress)

        return results

    async def abort_scan(self):
        """
        Abort currently running scan.

        The scan will stop gracefully after the current point completes
        and return partial results.

        Examples
        --------
        >>> # In one cell:
        >>> task = asyncio.create_task(
        ...     server.scan_from_dataframe(large_scan_df)
        ... )
        >>>
        >>> # In another cell while scan is running:
        >>> await server.abort_scan()
        """
        await self._scan_executor.abort()

    @property
    def is_scanning(self) -> bool:
        """
        Check if a scan is currently running.

        Returns
        -------
        bool
            True if scan is in progress
        """
        return self._scan_executor.current_scan is not None
