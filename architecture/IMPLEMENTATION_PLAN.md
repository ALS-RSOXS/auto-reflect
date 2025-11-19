# RSoXS Beamline API Implementation Plan

**Date:** November 5, 2025
**Project:** High-Level User-Friendly Interface for ALS RSoXS Beamline
**Repository:** auto-reflect (ALS-RSOXS)

---

## Executive Summary

This plan outlines the implementation of a safe, user-friendly Python interface for the ALS RSoXS beamline that:
1. Prevents race conditions through proper async/await patterns
2. Accepts pandas DataFrames as scan definitions
3. Uses `uncertainties.ufloat` throughout for automatic error propagation
4. Provides both low-level safe primitives and high-level scan orchestration
5. Implements comprehensive error handling with automatic recovery

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  (Jupyter Notebooks, High-Level Scan Functions)             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              High-Level API (RsoxsServer)                    │
│  • scan_from_dataframe()                                     │
│  • nexafs_scan()                                             │
│  • alignment_scan()                                          │
│  • Abort mechanism                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│         Mid-Level Orchestration (Scan Execution)             │
│  • Multi-motor coordination                                  │
│  • Progress tracking                                         │
│  • Error recovery                                            │
│  • Result aggregation with ufloat                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│        Low-Level Safe Primitives (Core Module)               │
│  • Async context managers (MotorMove, ShutterControl)       │
│  • Wait utilities (wait_for_motors, wait_for_settle)        │
│  • Resource cleanup                                          │
│  • Thread-safe abort flags                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           Type-Safe Accessors (RsoxsAccessor)                │
│  • AI accessor → ufloat values                              │
│  • Motor accessor → position tracking                        │
│  • DIO accessor → shutter control                           │
│  • Validation against type literals                         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                 BCSz API (Existing)                          │
│  • Low-level ZMQ communication                               │
│  • Motor commands, AI acquisition, DIO control              │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
api_dev/
├── src/
│   └── api_dev/
│       ├── __init__.py                 # Package initialization
│       ├── types.py                    # Type definitions, literals, exceptions
│       ├── core.py                     # Low-level safe async primitives
│       ├── server.py                   # RsoxsServer (main interface)
│       ├── accessors.py                # RsoxsAccessor with ufloat support
│       ├── scan.py                     # Scan orchestration logic
│       ├── validation.py               # DataFrame validation utilities
│       ├── nexafs.py                   # NEXAFS-specific functionality
│       └── utils.py                    # Helper functions
├── tests/
│   ├── test_validation.py
│   ├── test_core.py
│   ├── test_scan.py
│   └── test_nexafs.py
├── notebooks/
│   ├── 01_basic_operations.ipynb       # Tutorial: basic AI/motor/DIO
│   ├── 02_simple_scans.ipynb           # Tutorial: single motor scans
│   ├── 03_dataframe_scans.ipynb        # Tutorial: DataFrame-based scans
│   ├── 04_nexafs_workflow.ipynb        # Tutorial: complete NEXAFS
│   └── test_API.ipynb                  # Refactored working tests
├── IMPLEMENTATION_PLAN.md              # This document
├── architecture.md                     # Original architecture notes
├── README.md
└── pyproject.toml
```

---

## Phase 1: Core Type System and Error Classes

### 1.1 Custom Exceptions (`types.py`)

```python
class RsoxsError(Exception):
    """Base exception for RSoXS operations"""
    pass

class MotorError(RsoxsError):
    """Motor operation failed"""
    pass

class MotorTimeoutError(MotorError):
    """Motor move did not complete in time"""
    pass

class ShutterError(RsoxsError):
    """Shutter operation failed"""
    pass

class AcquisitionError(RsoxsError):
    """Data acquisition failed"""
    pass

class ValidationError(RsoxsError):
    """Scan plan validation failed"""
    pass

class ScanAbortedError(RsoxsError):
    """Scan was aborted by user"""
    pass
```

### 1.2 Core Data Structures (`types.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import uncertainties as un
from uncertainties import ufloat

@dataclass
class ScanPoint:
    """Single point in a scan trajectory"""
    index: int
    motors: Dict[Motor, float]  # Motor names -> positions
    exposure_time: float
    ai_channels: Optional[List[AI]] = None
    delay_after_move: float = 0.2

    def validate(self) -> None:
        """Validate motor names and exposure time"""
        if self.exposure_time <= 0:
            raise ValidationError(f"Invalid exposure time: {self.exposure_time}")
        # Additional validation...

@dataclass
class ScanResult:
    """Results from a scan point with uncertainty"""
    index: int
    motors: Dict[Motor, float]
    ai_data: Dict[AI, ufloat]  # Channel name -> ufloat value
    exposure_time: float
    timestamp: float
    raw_data: Dict[AI, List[float]]  # For debugging

    def to_series(self) -> pd.Series:
        """Convert to pandas Series with proper column names"""
        data = {}
        # Motor positions
        for motor, pos in self.motors.items():
            data[f"{motor}_position"] = pos
        # AI data with mean and std
        for chan, uval in self.ai_data.items():
            data[f"{chan}_mean"] = uval.nominal_value
            data[f"{chan}_std"] = uval.std_dev
        data["exposure"] = self.exposure_time
        data["timestamp"] = self.timestamp
        return pd.Series(data)
```

### 1.3 Column Name Mapping (`validation.py`)

```python
from typing import Optional
import re

EXPOSURE_PATTERNS = [
    r"^exposure$",
    r"^exp$",
    r"^count[_\s]?time",
    r"^integration[_\s]?time",
    r"^unnamed.*",  # Pandas unnamed column pattern
]

def find_exposure_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find exposure time column using pattern matching.

    Returns the column name if found, None otherwise.
    """
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for pattern in EXPOSURE_PATTERNS:
            if re.match(pattern, col_lower):
                return col
    return None

def validate_motor_columns(df: pd.DataFrame) -> List[str]:
    """
    Validate that DataFrame columns match known motor names.

    Returns list of valid motor column names.
    Raises ValidationError if invalid columns found.
    """
    valid_motors = get_args(Motor.__value__)
    motor_cols = []
    exposure_col = find_exposure_column(df)

    for col in df.columns:
        if col == exposure_col:
            continue
        if col not in valid_motors:
            raise ValidationError(
                f"Column '{col}' is not a valid motor name. "
                f"Valid motors: {valid_motors[:5]}... (see types.py for full list)"
            )
        motor_cols.append(col)

    if not motor_cols:
        raise ValidationError("DataFrame must contain at least one motor column")

    return motor_cols
```

---

## Phase 2: Low-Level Safe Async Primitives

### 2.1 Async Context Managers (`core.py`)

```python
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import asyncio
import time

class AbortFlag:
    """Thread-safe abort flag for scan cancellation"""
    def __init__(self):
        self._aborted = False
        self._lock = asyncio.Lock()

    async def set(self):
        async with self._lock:
            self._aborted = True

    async def is_set(self) -> bool:
        async with self._lock:
            return self._aborted

    async def clear(self):
        async with self._lock:
            self._aborted = False

@asynccontextmanager
async def motor_move(
    server: BCSz.BCSServer,
    motors: Dict[Motor, float],
    timeout: float = 30.0,
    backlash: bool = True
):
    """
    Context manager for safe motor movements with automatic position restoration.

    Usage:
        async with motor_move(server, {"Sample X": 10.0}) as initial_pos:
            # Do work at new position
            pass
        # Motors automatically return to initial_pos on exit or exception
    """
    # Record initial positions
    initial_response = await server.get_motor(motors=list(motors.keys()))
    initial_pos = {
        m["motor"]: m["position"]
        for m in initial_response["data"]
    }

    try:
        # Move to target positions
        command = "Backlash Move" if backlash else "Normal Move"
        await server.command_motor(
            commands=[command] * len(motors),
            motors=list(motors.keys()),
            goals=list(motors.values())
        )

        # Wait for completion
        await wait_for_motors(
            server,
            list(motors.keys()),
            timeout=timeout
        )

        yield initial_pos

    except Exception as e:
        raise MotorError(f"Motor move failed: {e}") from e

    finally:
        # Return to initial positions
        try:
            await server.command_motor(
                commands=["Backlash Move"] * len(initial_pos),
                motors=list(initial_pos.keys()),
                goals=list(initial_pos.values())
            )
            await wait_for_motors(
                server,
                list(initial_pos.keys()),
                timeout=timeout
            )
        except Exception as e:
            # Log but don't raise - we're in cleanup
            print(f"Warning: Failed to restore motor positions: {e}")

@asynccontextmanager
async def shutter_control(
    server: BCSz.BCSServer,
    shutter: DIO = "Light Output",
    delay_before_open: float = 0.2
):
    """
    Context manager for safe shutter control.
    Guarantees shutter closes even on exception.

    Usage:
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
```

### 2.2 Wait Utilities (`core.py`)

```python
async def wait_for_motors(
    server: BCSz.BCSServer,
    motors: List[Motor],
    timeout: float = 30.0,
    check_interval: float = 0.05,
    abort_flag: Optional[AbortFlag] = None
) -> None:
    """
    Wait for all motors to complete movement.

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

async def wait_for_settle(delay: float, abort_flag: Optional[AbortFlag] = None):
    """
    Wait for motor settling with abort check.

    Raises:
        ScanAbortedError: If abort_flag is set during wait
    """
    if delay <= 0:
        return

    steps = int(delay / 0.1)  # Check abort every 100ms
    for _ in range(steps):
        if abort_flag and await abort_flag.is_set():
            raise ScanAbortedError("Scan aborted during settle")
        await asyncio.sleep(0.1)

    # Sleep remainder
    remainder = delay - (steps * 0.1)
    if remainder > 0:
        await asyncio.sleep(remainder)
```

---

## Phase 3: Refactor RsoxsAccessor for ufloat

### 3.1 Updated AI Accessor (`accessors.py`)

```python
from uncertainties import ufloat
import numpy as np

class RsoxsAccessor(Generic[T]):
    """Type-safe accessor with ufloat support for AI channels"""

    # ... existing __init__, __getitem__, set methods ...

    async def get_with_uncertainty(
        self,
        keys: List[AI],
        acquisition_time: float = 1.0
    ) -> Dict[AI, ufloat]:
        """
        Acquire AI data and return as ufloat values.

        Returns:
            Dictionary mapping channel names to ufloat(mean, std_error)
        """
        if self.kind is not AI:
            raise TypeError("get_with_uncertainty only works with AI accessor")

        # Start acquisition
        await self.server.acquire_data(chans=keys, time=acquisition_time)

        # Get array data
        result = await self.server.get_acquired_array(chans=keys)

        # Calculate statistics
        uncertainty_data = {}
        for chan_data in result["chans"]:
            chan_name = chan_data["chan"]
            data = np.array(chan_data["data"], dtype=float)

            if len(data) == 0:
                uncertainty_data[chan_name] = ufloat(np.nan, np.nan)
            else:
                mean = np.nanmean(data)
                # Standard error of the mean
                std_err = np.nanstd(data, ddof=1) / np.sqrt(len(data))
                uncertainty_data[chan_name] = ufloat(mean, std_err)

        return uncertainty_data

    async def table_with_uncertainty(
        self,
        keys: List[AI],
        acquisition_time: float = 1.0
    ) -> pd.DataFrame:
        """
        Retrieve AI data as DataFrame with ufloat values.

        Returns DataFrame with columns for each channel containing ufloat objects.
        """
        if self.kind is not AI:
            raise TypeError("table_with_uncertainty only works with AI accessor")

        uncertainty_data = await self.get_with_uncertainty(keys, acquisition_time)

        # Create DataFrame with ufloat values
        df = pd.DataFrame([uncertainty_data])
        return df
```

---

## Phase 4: DataFrame Validation and Conversion

### 4.1 Scan Plan Builder (`scan.py`)

```python
from typing import List, Tuple
import pandas as pd

class ScanPlan:
    """Validated scan plan built from DataFrame"""

    def __init__(
        self,
        points: List[ScanPoint],
        motor_names: List[Motor],
        ai_channels: List[AI],
        shutter: DIO = "Light Output"
    ):
        self.points = points
        self.motor_names = motor_names
        self.ai_channels = ai_channels
        self.shutter = shutter

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        ai_channels: Optional[List[AI]] = None,
        default_exposure: float = 1.0,
        default_delay: float = 0.2,
        shutter: DIO = "Light Output"
    ) -> "ScanPlan":
        """
        Build validated scan plan from DataFrame.

        Parameters:
            df: DataFrame with motor columns and optional exposure column
            ai_channels: AI channels to read (None = all)
            default_exposure: Exposure time if no column found
            default_delay: Delay after motor move
            shutter: Shutter channel name

        Returns:
            Validated ScanPlan

        Raises:
            ValidationError: If DataFrame validation fails
        """
        # Find exposure column
        exposure_col = find_exposure_column(df)

        # Validate motor columns
        motor_cols = validate_motor_columns(df)

        # Build scan points
        points = []
        for idx, row in df.iterrows():
            motors = {col: row[col] for col in motor_cols}

            if exposure_col:
                exposure = row[exposure_col]
            else:
                exposure = default_exposure

            point = ScanPoint(
                index=idx,
                motors=motors,
                exposure_time=exposure,
                ai_channels=ai_channels,
                delay_after_move=default_delay
            )
            point.validate()
            points.append(point)

        return cls(
            points=points,
            motor_names=motor_cols,
            ai_channels=ai_channels or get_args(AI.__value__),
            shutter=shutter
        )

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)
```

---

## Phase 5: High-Level Scan Orchestration

### 5.1 Scan Executor (`scan.py`)

```python
from tqdm.asyncio import tqdm
import pandas as pd

class ScanExecutor:
    """Executes scan plans with progress tracking and error recovery"""

    def __init__(self, server: "RsoxsServer"):
        self.server = server
        self.abort_flag = AbortFlag()
        self.current_scan: Optional[ScanPlan] = None

    async def abort(self):
        """Request abort of current scan"""
        await self.abort_flag.set()

    async def execute_point(
        self,
        point: ScanPoint,
        motor_timeout: float = 30.0
    ) -> ScanResult:
        """
        Execute a single scan point with full error handling.

        Returns:
            ScanResult with ufloat values

        Raises:
            MotorError, ShutterError, AcquisitionError, ScanAbortedError
        """
        # Check abort
        if await self.abort_flag.is_set():
            raise ScanAbortedError("Scan aborted before point execution")

        # Move motors and restore on exception
        async with motor_move(
            self.server,
            point.motors,
            timeout=motor_timeout
        ) as initial_pos:

            # Wait for settling
            await wait_for_settle(point.delay_after_move, self.abort_flag)

            # Open shutter, collect data, close shutter
            async with shutter_control(
                self.server,
                shutter=self.current_scan.shutter,
                delay_before_open=0
            ):
                # Check abort before acquisition
                if await self.abort_flag.is_set():
                    raise ScanAbortedError("Scan aborted before acquisition")

                # Acquire data with uncertainty
                ai_data = await self.server.ai.get_with_uncertainty(
                    keys=point.ai_channels or self.current_scan.ai_channels,
                    acquisition_time=point.exposure_time
                )

                # Get raw data for debugging
                raw_response = await self.server.get_acquired_array(
                    chans=list(ai_data.keys())
                )
                raw_data = {
                    chan["chan"]: chan["data"]
                    for chan in raw_response["chans"]
                }

        # Create result
        return ScanResult(
            index=point.index,
            motors=point.motors,
            ai_data=ai_data,
            exposure_time=point.exposure_time,
            timestamp=time.time(),
            raw_data=raw_data
        )

    async def execute_scan(
        self,
        scan_plan: ScanPlan,
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Execute complete scan plan.

        Parameters:
            scan_plan: Validated scan plan
            progress: Show progress bar

        Returns:
            DataFrame with ufloat columns for AI data

        Raises:
            Various scan errors
        """
        self.current_scan = scan_plan
        await self.abort_flag.clear()

        results = []

        # Create progress bar
        iterator = tqdm(
            scan_plan.points,
            desc="Scanning",
            disable=not progress
        ) if progress else scan_plan.points

        try:
            for point in iterator:
                result = await self.execute_point(point)
                results.append(result)

        except ScanAbortedError:
            print(f"Scan aborted after {len(results)} points")
            if not results:
                raise

        finally:
            self.current_scan = None

        # Convert to DataFrame
        df = pd.DataFrame([r.to_series() for r in results])
        return df
```

### 5.2 Updated RsoxsServer (`server.py`)

```python
class RsoxsServer(BCSz.BCSServer):
    """High-level RSoXS beamline interface"""

    CONFIG = Connection()

    def __init__(self):
        super().__init__()

        # Type-safe accessors
        self.ai = RsoxsAccessor[AI](self, AI, readonly=True)
        self.motor = RsoxsAccessor[Motor](self, Motor)
        self.dio = RsoxsAccessor[DIO](self, DIO)

        # Scan executor
        self._scan_executor = ScanExecutor(self)

        # CCD state (deferred)
        self.ccd_ready: bool = False

    # ... existing connect_with_env, create methods ...

    async def scan_from_dataframe(
        self,
        df: pd.DataFrame,
        ai_channels: Optional[List[AI]] = None,
        default_exposure: float = 1.0,
        default_delay: float = 0.2,
        shutter: DIO = "Light Output",
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Execute scan from DataFrame definition.

        Parameters:
            df: DataFrame with motor position columns and optional exposure column
            ai_channels: AI channels to acquire (None = all displayed channels)
            default_exposure: Exposure time if not in DataFrame
            default_delay: Settling time after motor moves
            shutter: Shutter channel to control
            progress: Show progress bar

        Returns:
            DataFrame with results including ufloat AI data

        Example:
            >>> scan_df = pd.DataFrame({
            ...     "Sample X": [10, 11, 12],
            ...     "Sample Y": [0, 0, 0],
            ...     "exposure": [1.0, 1.5, 2.0]
            ... })
            >>> results = await server.scan_from_dataframe(scan_df)
            >>> results["Photodiode_mean"]  # Access nominal values
            >>> results["Photodiode_std"]   # Access uncertainties
        """
        # Build scan plan with validation
        scan_plan = ScanPlan.from_dataframe(
            df=df,
            ai_channels=ai_channels,
            default_exposure=default_exposure,
            default_delay=default_delay,
            shutter=shutter
        )

        # Execute scan
        return await self._scan_executor.execute_scan(
            scan_plan=scan_plan,
            progress=progress
        )

    async def abort_scan(self):
        """Abort currently running scan"""
        await self._scan_executor.abort()

    @property
    def is_scanning(self) -> bool:
        """Check if a scan is currently running"""
        return self._scan_executor.current_scan is not None
```

---

## Phase 6: NEXAFS-Specific Functionality

### 6.1 Energy Scans with Error Propagation (`nexafs.py`)

```python
from uncertainties import unumpy as unp
import pandas as pd

def calculate_nexafs(
    scan_data: pd.DataFrame,
    signal_channel: str = "TEY signal",
    normalization_channel: str = "AI 3 Izero",
    energy_motor: str = "Beamline Energy"
) -> pd.DataFrame:
    """
    Calculate NEXAFS absorption with automatic error propagation.

    Uses uncertainties package for proper error propagation through:
        μ = -ln(I/I0)

    Parameters:
        scan_data: DataFrame from scan_from_dataframe with _mean and _std columns
        signal_channel: Signal detector channel name
        normalization_channel: I0 channel name
        energy_motor: Energy motor name

    Returns:
        DataFrame with energy, absorption (ufloat), and intermediate values
    """
    # Reconstruct ufloat values from _mean and _std columns
    signal = unp.uarray(
        scan_data[f"{signal_channel}_mean"],
        scan_data[f"{signal_channel}_std"]
    )

    i0 = unp.uarray(
        scan_data[f"{normalization_channel}_mean"],
        scan_data[f"{normalization_channel}_std"]
    )

    energy = scan_data[f"{energy_motor}_position"]

    # Calculate transmission with error propagation
    transmission = signal / i0

    # Calculate absorption: μ = -ln(I/I0)
    # Filter out invalid values
    valid_mask = unp.nominal_values(transmission) > 1e-10

    absorption = unp.uarray(np.zeros_like(transmission), np.zeros_like(transmission))
    absorption[valid_mask] = -unp.log(transmission[valid_mask])

    # Build result DataFrame
    result = pd.DataFrame({
        "energy": energy,
        "absorption": absorption,  # ufloat array
        "absorption_mean": unp.nominal_values(absorption),
        "absorption_std": unp.std_devs(absorption),
        "transmission": transmission,  # ufloat array
        "transmission_mean": unp.nominal_values(transmission),
        "transmission_std": unp.std_devs(transmission),
        f"{signal_channel}": signal,
        f"{normalization_channel}": i0
    })

    # Drop invalid points
    result_clean = result[valid_mask].reset_index(drop=True)

    if len(result_clean) < len(result):
        print(f"Warning: Removed {len(result) - len(result_clean)} invalid data points")

    return result_clean

async def nexafs_scan(
    server: "RsoxsServer",
    energies: np.ndarray,
    exposure_time: float | np.ndarray = 1.0,
    signal_channel: str = "TEY signal",
    normalization_channel: str = "AI 3 Izero",
    energy_motor: str = "Beamline Energy",
    **kwargs
) -> pd.DataFrame:
    """
    High-level NEXAFS scan function.

    Parameters:
        server: Connected RsoxsServer instance
        energies: Array of energies to scan
        exposure_time: Single value or array of exposure times
        signal_channel: Signal detector
        normalization_channel: I0 detector
        energy_motor: Energy motor name
        **kwargs: Additional arguments for scan_from_dataframe

    Returns:
        DataFrame with calculated absorption and error bars
    """
    # Build scan DataFrame
    if isinstance(exposure_time, (int, float)):
        exposure_time = np.full_like(energies, exposure_time)

    scan_df = pd.DataFrame({
        energy_motor: energies,
        "exposure": exposure_time
    })

    # Execute scan
    raw_results = await server.scan_from_dataframe(
        df=scan_df,
        ai_channels=[signal_channel, normalization_channel],
        **kwargs
    )

    # Calculate NEXAFS
    nexafs_data = calculate_nexafs(
        raw_results,
        signal_channel=signal_channel,
        normalization_channel=normalization_channel,
        energy_motor=energy_motor
    )

    return nexafs_data
```

---

## Phase 7: Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_validation.py
def test_find_exposure_column():
    df = pd.DataFrame({"Sample X": [1, 2], "exposure": [1.0, 1.5]})
    assert find_exposure_column(df) == "exposure"

    df = pd.DataFrame({"Sample X": [1, 2], "exp": [1.0, 1.5]})
    assert find_exposure_column(df) == "exp"

    df = pd.DataFrame({"Sample X": [1, 2], "Unnamed: 2": [1.0, 1.5]})
    assert find_exposure_column(df) == "Unnamed: 2"

def test_validate_motor_columns():
    df = pd.DataFrame({"Sample X": [1, 2], "Sample Y": [0, 0]})
    cols = validate_motor_columns(df)
    assert "Sample X" in cols
    assert "Sample Y" in cols

    df = pd.DataFrame({"Invalid Motor": [1, 2]})
    with pytest.raises(ValidationError):
        validate_motor_columns(df)

# tests/test_core.py
@pytest.mark.asyncio
async def test_motor_move_context_manager(mock_server):
    """Test motor position restoration on exception"""
    initial_pos = {"Sample X": 0.0}
    target_pos = {"Sample X": 10.0}

    with pytest.raises(RuntimeError):
        async with motor_move(mock_server, target_pos) as init_pos:
            assert init_pos == initial_pos
            raise RuntimeError("Simulated error")

    # Verify motors returned to initial position
    # (check mock_server calls)

@pytest.mark.asyncio
async def test_shutter_closes_on_exception(mock_server):
    """Test shutter closes even on exception"""
    with pytest.raises(RuntimeError):
        async with shutter_control(mock_server):
            raise RuntimeError("Simulated error")

    # Verify shutter was closed
    # (check mock_server.set_do was called with False)

# tests/test_scan.py
@pytest.mark.asyncio
async def test_scan_abort(mock_server):
    """Test scan can be aborted mid-execution"""
    executor = ScanExecutor(mock_server)

    # Create scan plan
    df = pd.DataFrame({"Sample X": range(100)})
    plan = ScanPlan.from_dataframe(df)

    # Start scan in background
    scan_task = asyncio.create_task(
        executor.execute_scan(plan, progress=False)
    )

    # Abort after short delay
    await asyncio.sleep(0.1)
    await executor.abort()

    # Verify ScanAbortedError raised
    with pytest.raises(ScanAbortedError):
        await scan_task
```

---

## Phase 8: Migration Path

### 8.1 Incremental Refactoring

1. **Phase 8.1**: Create new modules without breaking existing code
   - Add `types.py`, `core.py`, `validation.py` as new files
   - Keep existing `server_setup.ipynb` working

2. **Phase 8.2**: Refactor accessors in place
   - Update `RsoxsAccessor` to add ufloat methods
   - Keep existing methods for backward compatibility
   - Add deprecation warnings

3. **Phase 8.3**: Create new `RsoxsServer` class
   - Inherit from existing implementation
   - Add new high-level methods
   - Keep low-level API available

4. **Phase 8.4**: Update test notebooks
   - Create new `03_dataframe_scans.ipynb` with examples
   - Refactor `test_API.ipynb` to use new interface
   - Keep old notebooks as `test_API_legacy.ipynb`

### 8.2 Backward Compatibility

```python
# In RsoxsAccessor
async def __getitem__(self, key):
    """Original method - still works"""
    # ... existing implementation ...

async def get_with_uncertainty(self, keys):
    """New method with ufloat support"""
    # ... new implementation ...

# Provide both interfaces
```

---

## Implementation Timeline

### Week 1: Foundation
- [ ] Create `types.py` with exceptions and data structures
- [ ] Implement `validation.py` with column detection
- [ ] Write unit tests for validation

### Week 2: Core Primitives
- [ ] Implement `core.py` with context managers
- [ ] Add abort mechanism
- [ ] Write tests for async primitives

### Week 3: Accessor Refactoring
- [ ] Add ufloat support to `RsoxsAccessor`
- [ ] Update `table()` method
- [ ] Test uncertainty calculations

### Week 4: Scan Orchestration
- [ ] Implement `ScanPlan` and `ScanExecutor`
- [ ] Add progress tracking
- [ ] Test error recovery

### Week 5: High-Level API
- [ ] Add `scan_from_dataframe()` to `RsoxsServer`
- [ ] Implement abort mechanism
- [ ] Integration tests

### Week 6: NEXAFS Support
- [ ] Implement `calculate_nexafs()` with error propagation
- [ ] Add `nexafs_scan()` convenience function
- [ ] Test on real data

### Week 7: Testing and Documentation
- [ ] Complete test suite
- [ ] Write tutorial notebooks
- [ ] Update README and architecture docs

### Week 8: Polish and Optimization
- [ ] Performance optimization
- [ ] Error message improvements
- [ ] Code review and cleanup

---

## Example Usage

### Basic DataFrame Scan

```python
import pandas as pd
from api_dev.server import RsoxsServer

# Connect to beamline
server = await RsoxsServer.create()

# Define scan as DataFrame
scan_plan = pd.DataFrame({
    "Sample X": [10.0, 10.5, 11.0, 11.5, 12.0],
    "Sample Y": [0.0, 0.0, 0.0, 0.0, 0.0],
    "exposure": [1.0, 1.0, 1.5, 1.5, 2.0]
})

# Execute scan
results = await server.scan_from_dataframe(
    df=scan_plan,
    ai_channels=["Photodiode", "TEY signal"],
    shutter="Light Output"
)

# Results have ufloat columns
print(results["Photodiode_mean"])  # Nominal values
print(results["Photodiode_std"])   # Uncertainties

# Automatic error propagation in calculations
transmission = results["Photodiode_mean"] / results["TEY signal_mean"]
```

### NEXAFS Scan

```python
import numpy as np
from api_dev.nexafs import nexafs_scan

# Define energy range
energies = np.linspace(280, 320, 200)  # Carbon K-edge

# Run scan with automatic absorption calculation
nexafs_results = await nexafs_scan(
    server=server,
    energies=energies,
    exposure_time=1.0,
    signal_channel="TEY signal",
    normalization_channel="AI 3 Izero"
)

# Plot with error bars
nexafs_results.plot(
    x="energy",
    y="absorption_mean",
    yerr="absorption_std"
)
```

### Abort Mechanism

```python
# In one cell:
scan_task = asyncio.create_task(
    server.scan_from_dataframe(large_scan_df)
)

# In another cell (while scan is running):
await server.abort_scan()

# Scan will abort gracefully and return partial results
```

---

## Open Questions and Future Enhancements

### Integration with `asyncstdlib`

Explore using `asyncstdlib` for:
- `asyncstdlib.zip()` for parallel motor moves
- `asyncstdlib.enumerate()` for async iteration
- `asyncstdlib.chain()` for combining scan segments

Example:
```python
import asyncstdlib as a

async for point, result in a.zip(scan_points, result_stream):
    # Process scan points as they complete
    pass
```

### Feedback-Driven Scans

Future enhancement for adaptive scanning:
```python
async def adaptive_scan(
    server: RsoxsServer,
    initial_points: np.ndarray,
    condition: Callable[[ScanResult], bool],
    refine: Callable[[List[ScanResult]], np.ndarray]
):
    """Scan with adaptive point selection based on results"""
    # Implementation for future phase
    pass
```

### Auto-Alignment

Future enhancement for automatic sample alignment:
```python
async def auto_align(
    server: RsoxsServer,
    motor: Motor,
    signal_channel: AI,
    scan_range: Tuple[float, float]
):
    """Automatically find peak position for alignment"""
    # Implementation for future phase
    pass
```

---

## Dependencies

### Required Packages

```toml
[project]
dependencies = [
    "pandas >= 2.0",
    "numpy >= 1.24",
    "uncertainties >= 3.1",
    "pydantic >= 2.0",
    "pydantic-settings >= 2.0",
    "tqdm >= 4.65",
    "matplotlib >= 3.7",
    "asyncio",
]

[project.optional-dependencies]
dev = [
    "pytest >= 7.0",
    "pytest-asyncio >= 0.21",
    "pytest-cov >= 4.0",
    "black >= 23.0",
    "ruff >= 0.1",
]
```

---

## Success Criteria

1. ✅ **Race Condition Prevention**: No motor/shutter timing issues
2. ✅ **DataFrame Interface**: Users can define scans in familiar format
3. ✅ **Uncertainty Propagation**: Automatic error propagation through calculations
4. ✅ **Error Recovery**: Motors return to initial positions on errors
5. ✅ **Abort Mechanism**: Scans can be safely aborted from Jupyter
6. ✅ **Type Safety**: Full type hints with IDE support
7. ✅ **Test Coverage**: >80% code coverage with unit and integration tests
8. ✅ **Documentation**: Tutorial notebooks for common workflows

---

## Notes

- **CCD integration** deferred until AI/motor/DIO system is solid
- **Feedback scans** and **auto-alignment** are future enhancements
- **Backward compatibility** maintained during migration
- **Progressive enhancement**: Low-level primitives usable directly for custom workflows

---

**End of Implementation Plan**
