"""Scan orchestration and execution"""

import time
from typing import List, Optional

import numpy as np
import pandas as pd
from bcs.BCSz import BCSServer
from uncertainties import ufloat

from .core import AbortFlag, motor_move, shutter_control, wait_for_settle
from .types import Instrument, ScanAbortedError, ScanPoint, ScanResult
from .validation import validate_scan_dataframe

# ============================================================================
# Scan Plan
# ============================================================================


class ScanPlan:
    """Validated scan plan built from DataFrame"""

    def __init__(
        self,
        points: List[ScanPoint],
        motor_names: List[str],
        ai_channels: List[str],
        shutter: str = "Light Output",
        instrument: Instrument = "Photodiode",
        actuate_every: bool = True,
    ):
        self.points = points
        self.motor_names = motor_names
        self.ai_channels = ai_channels
        self.shutter = shutter

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        ai_channels: Optional[List[str]] = None,
        default_exposure: float = 1.0,
        default_delay: float = 0.2,
        shutter: str = "Light Output",
        instrument: Instrument = "Photodiode",
    ) -> "ScanPlan":
        """
        Build validated scan plan from DataFrame.

        Parameters:
            df: DataFrame with motor columns and optional exposure column
            ai_channels: AI channels to read (None = will use defaults)
            default_exposure: Exposure time if no column found
            default_delay: Delay after motor move
            shutter: Shutter channel name

        Returns:
            Validated ScanPlan

        Raises:
            ValidationError: If DataFrame validation fails
        """
        # Validate DataFrame
        motor_cols, exposure_col = validate_scan_dataframe(df)

        # Use default AI channels if not specified
        if ai_channels is None:
            # Default to common channels
            # TODO: @tjferron - maybe we should make this configurable? or a larger set?
            ai_channels = ["Photodiode", "TEY signal", "AI 3 Izero"]

        # Build scan points
        points = []
        for idx, row in df.iterrows():
            motors = {col: float(row[col]) for col in motor_cols}

            if exposure_col:
                exposure = float(row[exposure_col])
            else:
                exposure = default_exposure

            point = ScanPoint(
                index=int(idx),
                motors=motors,
                exposure_time=exposure,
                ai_channels=ai_channels,
                delay_after_move=default_delay,
            )
            point.validate()
            points.append(point)

        return cls(
            points=points,
            motor_names=motor_cols,
            ai_channels=ai_channels,
            shutter=shutter,
        )

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)


# ============================================================================
# Scan Executor
# ============================================================================


class ScanExecutor:
    """Executes scan plans with progress tracking and error recovery"""

    def __init__(self, server: BCSServer):
        self.server = server
        self.abort_flag = AbortFlag()
        self.current_scan: Optional[ScanPlan] = None

    async def abort(self):
        """Request abort of current scan"""
        await self.abort_flag.set()

    async def execute_point(
        self,
        point: ScanPoint,
        motor_timeout: float = 30.0,
        restore_motors: bool = False,
    ) -> ScanResult:
        """
        Execute a single scan point with full error handling.

        Parameters:
            point: Scan point to execute
            motor_timeout: Timeout for motor moves
            restore_motors: Whether to restore motor positions after point

        Returns:
            ScanResult with ufloat values

        Raises:
            MotorError, ShutterError, AcquisitionError, ScanAbortedError
        """
        # Check abort
        if await self.abort_flag.is_set():
            raise ScanAbortedError("Scan aborted before point execution")

        # Move motors (optionally restore on exit)
        async with motor_move(
            self.server,
            point.motors,
            timeout=motor_timeout,
            restore_on_exit=restore_motors,
        ) as initial_pos:  # noqa: F841
            # Wait for settling
            await wait_for_settle(point.delay_after_move, self.abort_flag)

            # Open shutter, collect data, close shutter
            async with shutter_control(
                self.server, shutter=self.current_scan.shutter, delay_before_open=0
            ):
                # Check abort before acquisition
                if await self.abort_flag.is_set():
                    raise ScanAbortedError("Scan aborted before acquisition")

                # Acquire data
                await self.server.acquire_data(
                    chans=point.ai_channels or self.current_scan.ai_channels,
                    time=point.exposure_time,
                )

                # Get array data
                result = await self.server.get_acquired_array(
                    chans=point.ai_channels or self.current_scan.ai_channels
                )

                # Calculate statistics with uncertainty
                ai_data = {}
                raw_data = {}

                for chan_data in result["chans"]:
                    chan_name = chan_data["chan"]
                    data = np.array(chan_data["data"], dtype=float)
                    raw_data[chan_name] = data.tolist()

                    if len(data) == 0:
                        ai_data[chan_name] = ufloat(np.nan, np.nan)
                    else:
                        mean = np.nanmean(data)
                        # Standard error of the mean
                        std_err = np.nanstd(data, ddof=1) / np.sqrt(len(data))
                        ai_data[chan_name] = ufloat(mean, std_err)

        # Create result
        return ScanResult(
            index=point.index,
            motors=point.motors,
            ai_data=ai_data,
            exposure_time=point.exposure_time,
            timestamp=time.time(),
            raw_data=raw_data,
        )

    async def execute_scan(
        self, scan_plan: ScanPlan, progress: bool = True
    ) -> pd.DataFrame:
        """
        Execute complete scan plan.

        Parameters:
            scan_plan: Validated scan plan
            progress: Show progress bar (requires tqdm)

        Returns:
            DataFrame with mean and std columns for AI data

        Raises:
            Various scan errors
        """
        self.current_scan = scan_plan
        await self.abort_flag.clear()

        results = []

        # Create progress indicator
        if progress:
            try:
                from tqdm.asyncio import tqdm

                iterator = tqdm(scan_plan.points, desc="Scanning", unit="pt")
            except ImportError:
                print("tqdm not available, showing simple progress")
                iterator = scan_plan.points
                progress = False
        else:
            iterator = scan_plan.points

        try:
            for i, point in enumerate(iterator):
                if not progress:
                    print(f"Point {i + 1}/{len(scan_plan.points)}", end="\r")

                result = await self.execute_point(point, restore_motors=False)
                results.append(result)

        except ScanAbortedError:
            print(f"\nScan aborted after {len(results)}/{len(scan_plan.points)} points")
            if not results:
                raise

        finally:
            self.current_scan = None
            if not progress:
                print()  # New line after progress

        # Convert to DataFrame
        df = pd.DataFrame([r.to_series() for r in results])
        return df
