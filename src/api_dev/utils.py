"""
Utility functions for beamline operations.

This module provides helper functions for common beamline tasks
like motor alignment, grid scan generation, and data analysis.
"""

import numpy as np
import pandas as pd


def create_grid_scan(
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
    exposure_time: float = 1.0,
    x_motor: str = "Sample X",
    y_motor: str = "Sample Y",
) -> pd.DataFrame:
    """
    Create a 2D grid scan DataFrame.

    Parameters
    ----------
    x_range : tuple[float, float, int]
        X range as (start, stop, num_points)
    y_range : tuple[float, float, int]
        Y range as (start, stop, num_points)
    exposure_time : float, optional
        Exposure time for all points (default: 1.0)
    x_motor : str, optional
        X motor name (default: "Sample X")
    y_motor : str, optional
        Y motor name (default: "Sample Y")

    Returns
    -------
    pd.DataFrame
        Scan definition ready for scan_from_dataframe

    Examples
    --------
    >>> # Create 3x3 grid
    >>> grid = create_grid_scan(
    ...     x_range=(10, 12, 3),
    ...     y_range=(0, 2, 3),
    ...     exposure_time=1.0
    ... )
    >>> print(len(grid))  # 9 points
    9
    """
    x_positions = np.linspace(x_range[0], x_range[1], x_range[2])
    y_positions = np.linspace(y_range[0], y_range[1], y_range[2])

    X, Y = np.meshgrid(x_positions, y_positions)

    return pd.DataFrame(
        {
            x_motor: X.flatten(),
            y_motor: Y.flatten(),
            "exposure": np.full(X.size, exposure_time),
        }
    )


def create_line_scan(
    motor: str,
    start: float,
    stop: float,
    num_points: int,
    exposure_time: float = 1.0,
) -> pd.DataFrame:
    """
    Create a 1D line scan DataFrame.

    Parameters
    ----------
    motor : str
        Motor name
    start : float
        Start position
    stop : float
        Stop position
    num_points : int
        Number of points
    exposure_time : float, optional
        Exposure time for all points (default: 1.0)

    Returns
    -------
    pd.DataFrame
        Scan definition ready for scan_from_dataframe

    Examples
    --------
    >>> scan = create_line_scan(
    ...     motor="Sample X",
    ...     start=10,
    ...     stop=15,
    ...     num_points=11,
    ...     exposure_time=1.5
    ... )
    """
    positions = np.linspace(start, stop, num_points)
    return pd.DataFrame(
        {motor: positions, "exposure": np.full(num_points, exposure_time)}
    )


def create_energy_scan(
    energies: np.ndarray,
    exposure_time: float | np.ndarray = 1.0,
    energy_motor: str = "Beamline Energy",
) -> pd.DataFrame:
    """
    Create an energy scan DataFrame.

    Parameters
    ----------
    energies : np.ndarray
        Array of photon energies
    exposure_time : float | np.ndarray, optional
        Exposure time(s) (default: 1.0)
    energy_motor : str, optional
        Energy motor name (default: "Beamline Energy")

    Returns
    -------
    pd.DataFrame
        Scan definition ready for scan_from_dataframe

    Examples
    --------
    >>> # Carbon K-edge
    >>> energies = np.linspace(280, 320, 200)
    >>> scan = create_energy_scan(energies, exposure_time=1.0)
    >>>
    >>> # Variable exposure times
    >>> energies = np.linspace(280, 320, 200)
    >>> exposure = np.ones(200)
    >>> exposure[100:150] = 2.0  # Longer at edge
    >>> scan = create_energy_scan(energies, exposure_time=exposure)
    """
    if isinstance(exposure_time, (int, float)):
        exposure_time = np.full_like(energies, exposure_time, dtype=float)

    return pd.DataFrame({energy_motor: energies, "exposure": exposure_time})


def find_peak_position(
    scan_data: pd.DataFrame,
    motor_col: str,
    signal_col: str,
    use_mean: bool = True,
) -> tuple[float, float]:
    """
    Find peak position in 1D scan data.

    Parameters
    ----------
    scan_data : pd.DataFrame
        Scan results from scan_from_dataframe
    motor_col : str
        Motor position column name (e.g., "Sample X_position")
    signal_col : str
        Signal column name (e.g., "Photodiode_mean")
    use_mean : bool, optional
        Use _mean column if True, else use raw column (default: True)

    Returns
    -------
    tuple[float, float]
        (peak_position, peak_value)

    Examples
    --------
    >>> results = await server.scan_from_dataframe(scan_df, ...)
    >>> peak_x, peak_signal = find_peak_position(
    ...     results,
    ...     motor_col="Sample X_position",
    ...     signal_col="Photodiode_mean"
    ... )
    >>> print(f"Peak at {peak_x:.2f} with signal {peak_signal:.3f}")
    """
    if use_mean and not signal_col.endswith("_mean"):
        signal_col = f"{signal_col}_mean"

    peak_idx = scan_data[signal_col].idxmax()
    peak_position = scan_data.loc[peak_idx, motor_col]
    peak_value = scan_data.loc[peak_idx, signal_col]

    return peak_position, peak_value


def calculate_center_of_mass(
    scan_data: pd.DataFrame,
    motor_col: str,
    signal_col: str,
    use_mean: bool = True,
) -> float:
    """
    Calculate center of mass for alignment.

    Parameters
    ----------
    scan_data : pd.DataFrame
        Scan results from scan_from_dataframe
    motor_col : str
        Motor position column name
    signal_col : str
        Signal column name
    use_mean : bool, optional
        Use _mean column if True (default: True)

    Returns
    -------
    float
        Center of mass position

    Examples
    --------
    >>> results = await server.scan_from_dataframe(scan_df, ...)
    >>> com = calculate_center_of_mass(
    ...     results,
    ...     motor_col="Sample X_position",
    ...     signal_col="Photodiode_mean"
    ... )
    >>> # Move to center of mass
    >>> await server.motor.set("Sample X", com)
    """
    if use_mean and not signal_col.endswith("_mean"):
        signal_col = f"{signal_col}_mean"

    positions = scan_data[motor_col].values
    signal = scan_data[signal_col].values

    # Ensure signal is positive
    signal = signal - signal.min()

    if signal.sum() == 0:
        raise ValueError("Signal is zero or negative everywhere")

    com = np.sum(positions * signal) / np.sum(signal)
    return com


def resample_scan_data(
    scan_data: pd.DataFrame,
    motor_col: str,
    num_points: int,
    columns_to_interpolate: list[str] | None = None,
) -> pd.DataFrame:
    """
    Resample scan data to uniform grid.

    Useful for combining scans with different point densities.

    Parameters
    ----------
    scan_data : pd.DataFrame
        Original scan data
    motor_col : str
        Motor position column for interpolation axis
    num_points : int
        Number of points in resampled data
    columns_to_interpolate : list[str] | None, optional
        Columns to interpolate (default: all numeric columns)

    Returns
    -------
    pd.DataFrame
        Resampled data with uniform spacing

    Examples
    --------
    >>> # Resample to 100 points
    >>> resampled = resample_scan_data(
    ...     results,
    ...     motor_col="Beamline Energy_position",
    ...     num_points=100
    ... )
    """
    from scipy.interpolate import interp1d

    if columns_to_interpolate is None:
        # Auto-detect numeric columns except motor position
        columns_to_interpolate = [
            col
            for col in scan_data.select_dtypes(include=[np.number]).columns
            if col != motor_col
        ]

    # Create uniform grid
    motor_positions = scan_data[motor_col].values
    new_positions = np.linspace(
        motor_positions.min(), motor_positions.max(), num_points
    )

    # Interpolate each column
    resampled_data = {motor_col: new_positions}

    for col in columns_to_interpolate:
        interpolator = interp1d(
            motor_positions,
            scan_data[col].values,
            kind="linear",
            fill_value="extrapolate",
        )
        resampled_data[col] = interpolator(new_positions)

    return pd.DataFrame(resampled_data)


def merge_scans(
    scans: list[pd.DataFrame],
    motor_col: str,
    average_overlaps: bool = True,
) -> pd.DataFrame:
    """
    Merge multiple scans into a single dataset.

    Parameters
    ----------
    scans : list[pd.DataFrame]
        List of scan DataFrames to merge
    motor_col : str
        Motor position column to use for alignment
    average_overlaps : bool, optional
        Average overlapping points (default: True)

    Returns
    -------
    pd.DataFrame
        Merged scan data

    Examples
    --------
    >>> # Combine multiple energy ranges
    >>> scan1 = await nexafs_scan(server, np.linspace(280, 290, 50))
    >>> scan2 = await nexafs_scan(server, np.linspace(288, 300, 100))
    >>> merged = merge_scans([scan1, scan2], motor_col="energy")
    """
    if not scans:
        raise ValueError("Need at least one scan to merge")

    # Concatenate all scans
    combined = pd.concat(scans, ignore_index=True)

    # Sort by motor position
    combined = combined.sort_values(motor_col).reset_index(drop=True)

    if average_overlaps:
        # Group by motor position and average
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined = combined.groupby(motor_col, as_index=False)[numeric_cols].mean()

    return combined
