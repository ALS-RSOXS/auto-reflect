"""
NEXAFS-specific functionality with automatic error propagation.

This module provides functions for calculating NEXAFS absorption spectra
with proper uncertainty propagation using the uncertainties package.
"""

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp


def calculate_nexafs(
    scan_data: pd.DataFrame,
    signal_channel: str = "TEY signal",
    normalization_channel: str = "AI 3 Izero",
    energy_motor: str = "Beamline Energy",
) -> pd.DataFrame:
    """
    Calculate NEXAFS absorption with automatic error propagation.

    Uses uncertainties package for proper error propagation through:
        μ = -ln(I/I0)

    where I is the signal (TEY, fluorescence, etc.) and I0 is the
    incident beam intensity.

    Parameters
    ----------
    scan_data : pd.DataFrame
        DataFrame from scan_from_dataframe with _mean and _std columns
    signal_channel : str, optional
        Signal channel name (default: "TEY signal")
    normalization_channel : str, optional
        Normalization channel name (default: "AI 3 Izero")
    energy_motor : str, optional
        Energy motor name (default: "Beamline Energy")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - energy: Photon energy values
        - transmission_mean/std: I/I0 with uncertainty
        - absorption_mean/std: -ln(I/I0) with uncertainty
        - {signal_channel}_mean/std: Raw signal with uncertainty
        - {normalization_channel}_mean/std: Raw I0 with uncertainty

    Examples
    --------
    >>> # After running energy scan
    >>> results = await server.scan_from_dataframe(energy_scan_df, ...)
    >>> nexafs = calculate_nexafs(
    ...     results,
    ...     signal_channel="TEY signal",
    ...     normalization_channel="AI 3 Izero"
    ... )
    >>> # Plot with error bars
    >>> import matplotlib.pyplot as plt
    >>> plt.errorbar(
    ...     nexafs["energy"],
    ...     nexafs["absorption_mean"],
    ...     yerr=nexafs["absorption_std"],
    ...     marker='o'
    ... )
    """
    # Reconstruct ufloat values from _mean and _std columns
    signal = unp.uarray(
        scan_data[f"{signal_channel}_mean"], scan_data[f"{signal_channel}_std"]
    )

    i0 = unp.uarray(
        scan_data[f"{normalization_channel}_mean"],
        scan_data[f"{normalization_channel}_std"],
    )

    energy = scan_data[f"{energy_motor}_position"]

    # Calculate transmission with error propagation
    transmission = signal / i0

    # Calculate absorption: μ = -ln(I/I0)
    # Filter out invalid values (transmission <= 0 or very small)
    valid_mask = unp.nominal_values(transmission) > 1e-10

    # Initialize absorption array
    absorption = unp.uarray(
        np.zeros_like(unp.nominal_values(transmission)),
        np.zeros_like(unp.std_devs(transmission)),
    )

    # Only calculate for valid transmission values
    if np.any(valid_mask):
        absorption[valid_mask] = -unp.log(transmission[valid_mask])

    # Build result DataFrame
    result = pd.DataFrame(
        {
            "energy": energy,
            "transmission_mean": unp.nominal_values(transmission),
            "transmission_std": unp.std_devs(transmission),
            "absorption_mean": unp.nominal_values(absorption),
            "absorption_std": unp.std_devs(absorption),
            f"{signal_channel}_mean": unp.nominal_values(signal),
            f"{signal_channel}_std": unp.std_devs(signal),
            f"{normalization_channel}_mean": unp.nominal_values(i0),
            f"{normalization_channel}_std": unp.std_devs(i0),
        }
    )

    # Drop invalid points
    result_clean = result[valid_mask].reset_index(drop=True)

    if len(result_clean) < len(result):
        n_removed = len(result) - len(result_clean)
        print(f"Warning: Removed {n_removed} invalid data points (transmission <= 0)")

    return result_clean


async def nexafs_scan(
    server,
    energies: np.ndarray,
    exposure_time: float | np.ndarray = 1.0,
    signal_channel: str = "TEY signal",
    normalization_channel: str = "AI 3 Izero",
    energy_motor: str = "Beamline Energy",
    **kwargs,
) -> pd.DataFrame:
    """
    High-level NEXAFS scan function with automatic absorption calculation.

    This function:
    1. Creates a scan DataFrame from energy array
    2. Executes the scan with specified exposure times
    3. Calculates absorption with error propagation
    4. Returns clean DataFrame with uncertainties

    Parameters
    ----------
    server : RsoxsServer
        Connected RsoxsServer instance
    energies : np.ndarray
        Array of photon energies to scan
    exposure_time : float | np.ndarray, optional
        Exposure time(s) in seconds (default: 1.0)
        Can be single value or array matching energies length
    signal_channel : str, optional
        Signal channel name (default: "TEY signal")
    normalization_channel : str, optional
        I0 channel name (default: "AI 3 Izero")
    energy_motor : str, optional
        Energy motor name (default: "Beamline Energy")
    **kwargs
        Additional arguments passed to scan_from_dataframe
        (e.g., default_delay, shutter, motor_timeout)

    Returns
    -------
    pd.DataFrame
        NEXAFS data with calculated absorption and error bars

    Examples
    --------
    >>> import numpy as np
    >>> from api_dev import nexafs_scan
    >>>
    >>> # Carbon K-edge scan
    >>> energies = np.linspace(280, 320, 200)
    >>> nexafs_data = await nexafs_scan(
    ...     server,
    ...     energies=energies,
    ...     exposure_time=1.0,
    ...     signal_channel="TEY signal",
    ...     normalization_channel="AI 3 Izero"
    ... )
    >>>
    >>> # Plot with error bars
    >>> import matplotlib.pyplot as plt
    >>> plt.errorbar(
    ...     nexafs_data["energy"],
    ...     nexafs_data["absorption_mean"],
    ...     yerr=nexafs_data["absorption_std"],
    ...     marker='o',
    ...     capsize=3
    ... )
    >>> plt.xlabel("Energy (eV)")
    >>> plt.ylabel("Absorption (a.u.)")
    >>> plt.title("Carbon K-edge NEXAFS")
    """
    # Build scan DataFrame
    if isinstance(exposure_time, (int, float)):
        exposure_time = np.full_like(energies, exposure_time, dtype=float)

    scan_df = pd.DataFrame({energy_motor: energies, "exposure": exposure_time})

    # Execute scan
    raw_results = await server.scan_from_dataframe(
        df=scan_df,
        ai_channels=[signal_channel, normalization_channel],
        **kwargs,
    )

    # Calculate NEXAFS
    nexafs_data = calculate_nexafs(
        raw_results,
        signal_channel=signal_channel,
        normalization_channel=normalization_channel,
        energy_motor=energy_motor,
    )

    return nexafs_data


def normalize_to_edge_jump(
    nexafs_data: pd.DataFrame,
    pre_edge_range: tuple[float, float],
    post_edge_range: tuple[float, float],
) -> pd.DataFrame:
    """
    Normalize NEXAFS data to edge jump.

    Fits linear baselines to pre-edge and post-edge regions,
    then normalizes so that edge jump = 1.

    Parameters
    ----------
    nexafs_data : pd.DataFrame
        NEXAFS data from calculate_nexafs or nexafs_scan
    pre_edge_range : tuple[float, float]
        Energy range (min, max) for pre-edge baseline
    post_edge_range : tuple[float, float]
        Energy range (min, max) for post-edge baseline

    Returns
    -------
    pd.DataFrame
        Normalized NEXAFS data with additional columns:
        - absorption_normalized_mean
        - absorption_normalized_std

    Examples
    --------
    >>> normalized = normalize_to_edge_jump(
    ...     nexafs_data,
    ...     pre_edge_range=(280, 282),
    ...     post_edge_range=(310, 315)
    ... )
    """
    energy = nexafs_data["energy"].values
    absorption = unp.uarray(
        nexafs_data["absorption_mean"], nexafs_data["absorption_std"]
    )

    # Find pre-edge and post-edge regions
    pre_mask = (energy >= pre_edge_range[0]) & (energy <= pre_edge_range[1])
    post_mask = (energy >= post_edge_range[0]) & (energy <= post_edge_range[1])

    # Calculate average values with uncertainty
    # Use sum/len instead of mean for ufloat arrays
    pre_edge_avg = absorption[pre_mask].sum() / pre_mask.sum()
    post_edge_avg = absorption[post_mask].sum() / post_mask.sum()

    # Edge jump
    edge_jump = post_edge_avg - pre_edge_avg

    # Normalize
    normalized = (absorption - pre_edge_avg) / edge_jump

    # Add to DataFrame
    result = nexafs_data.copy()
    result["absorption_normalized_mean"] = unp.nominal_values(normalized)
    result["absorption_normalized_std"] = unp.std_devs(normalized)

    return result
