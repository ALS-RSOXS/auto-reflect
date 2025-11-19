"""
RSoXS Beamline Control API

A high-level, user-friendly Python interface for the ALS RSoXS beamline that:
- Prevents race conditions through proper async/await patterns
- Accepts pandas DataFrames as scan definitions
- Uses uncertainties.ufloat for automatic error propagation
- Provides safe primitives and high-level orchestration
- Implements comprehensive error handling with automatic recovery
"""

__version__ = "0.1.0"

# Core types and exceptions
# Type-safe accessors
from .accessors import (
    RsoxsAccessor,
    TabularResponse,
)

# Core async primitives
from .core import (
    AbortFlag,
    motor_move,
    shutter_control,
    wait_for_motors,
    wait_for_settle,
)

# NEXAFS functionality
from .nexafs import (
    calculate_nexafs,
    nexafs_scan,
    normalize_to_edge_jump,
)

# Scan orchestration
from .scan import (
    ScanExecutor,
    ScanPlan,
)

# High-level server
from .server import (
    Connection,
    RsoxsServer,
)
from .types import (
    # Type literals
    AI,
    DIO,
    AcquisitionError,
    Command,
    Motor,
    MotorError,
    MotorTimeoutError,
    # Exceptions
    RsoxsError,
    ScanAbortedError,
    # Data structures
    ScanPoint,
    ScanResult,
    ShutterError,
    ValidationError,
    # Type tuples for validation
    ai,
    command,
    dio,
    motor,
)

# Utility functions
from .utils import (
    calculate_center_of_mass,
    create_energy_scan,
    create_grid_scan,
    create_line_scan,
    find_peak_position,
    merge_scans,
    resample_scan_data,
)

# Validation utilities
from .validation import (
    find_exposure_column,
    validate_motor_columns,
    validate_scan_dataframe,
)

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "RsoxsError",
    "MotorError",
    "MotorTimeoutError",
    "ShutterError",
    "AcquisitionError",
    "ValidationError",
    "ScanAbortedError",
    # Data structures
    "ScanPoint",
    "ScanResult",
    "TabularResponse",
    # Types
    "AI",
    "Motor",
    "DIO",
    "Command",
    "ai",
    "motor",
    "dio",
    "command",
    # Validation
    "find_exposure_column",
    "validate_motor_columns",
    "validate_scan_dataframe",
    # Core primitives
    "AbortFlag",
    "wait_for_motors",
    "wait_for_settle",
    "motor_move",
    "shutter_control",
    # Scan
    "ScanPlan",
    "ScanExecutor",
    # Accessors and Server
    "RsoxsAccessor",
    "RsoxsServer",
    "Connection",
    # NEXAFS
    "calculate_nexafs",
    "nexafs_scan",
    "normalize_to_edge_jump",
    # Utilities
    "create_grid_scan",
    "create_line_scan",
    "create_energy_scan",
    "find_peak_position",
    "calculate_center_of_mass",
    "resample_scan_data",
    "merge_scans",
]
