"""
Type-safe accessors for beamline components with uncertainty support.

This module provides the RsoxsAccessor class as a Hardware Abstraction for accessing
AI channels, motors, and digital I/O with automatic uncertainty propagation.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, get_args

import numpy as np
import pandas as pd
from bcs import BCSz
from uncertainties import ufloat

from .types import AI, DIO, Motor

T = TypeVar("T", bound=AI | Motor | DIO)

# ===========================================================================
# Uncertainty Propagation Management
# ===========================================================================

# ==========================================================================
# Trajectories and Global Positions
# ==========================================================================


@dataclass
class TabularResponse:
    """Container for tabular data with state and status DataFrames."""

    state: pd.DataFrame
    status: pd.DataFrame


class RsoxsAccessor(Generic[T]):
    """
    Type-safe accessor for beamline components.

    Provides methods to get and set AI channels, motors, and digital I/O
    with automatic validation and uncertainty support.

    Parameters
    ----------
    server : BCSz.BCSServer
        Connected BCS server instance
    kind : type[T]
        Type literal (AI, Motor, or DIO)
    readonly : bool, optional
        If True, prevent set operations (default: False)

    Examples
    --------
    >>> server = await RsoxsServer.create()
    >>> ai_data = await server.ai.get_with_uncertainty(
    ...     keys=["Photodiode", "TEY signal"],
    ...     acquisition_time=1.0
    ... )
    >>> print(ai_data["Photodiode"])  # ufloat(mean, std_error)
    """

    def __init__(
        self, server: BCSz.BCSServer, kind: type[T], *, readonly: bool = False
    ) -> None:
        self.server = server
        self.kind = kind
        self.readonly = readonly

    async def __getitem__(self, key: T | tuple[T, ...]):
        """
        Get current value(s) for the specified key(s).

        Parameters
        ----------
        key : T | tuple[T, ...]
            Single key or tuple of keys to retrieve

        Returns
        -------
        dict
            Response dictionary from BCS server

        Raises
        ------
        KeyError
            If key is not valid for this accessor type
        """
        # Handle both single key and multiple keys
        if isinstance(key, str):
            keys = (key,)
        else:
            keys = key

        # Check if all keys are valid for this accessor type
        if any(k not in get_args(self.kind.__value__) for k in keys):
            raise KeyError(f"{keys} is not a valid {self.kind}")

        if self.kind is AI:
            result = await self.server.get_acquired_array(chans=list(keys))
            return result
        elif self.kind is Motor:
            result = await self.server.get_motor(motors=list(keys))
            return result
        elif self.kind is DIO:
            result = await self.server.get_di(chans=list(keys))
            return result
        else:
            raise NotImplementedError(
                f"Accessor for type {self.kind} is not implemented."
            )

    async def set(self, key: str, value):
        """
        Set value for the specified key.

        Parameters
        ----------
        key : str
            Component name to set
        value : float | bool
            Value to set (float for motors, bool for DIO)

        Raises
        ------
        PermissionError
            If accessor is read-only
        KeyError
            If key is not valid for this accessor type
        ValueError
            If value type is incorrect for DIO
        """
        if self.readonly:
            raise PermissionError("This accessor is read-only.")

        # Check if key is valid for this accessor type
        if key not in get_args(self.kind.__value__):
            raise KeyError(f"{key} is not a valid {self.kind}")

        if self.kind is Motor:
            await self.server.command_motor(
                commands=["Backlash Move"], motors=[key], goals=[value]
            )
        elif self.kind is DIO:
            if not isinstance(value, (int, bool)):
                raise ValueError("DIO value must be an integer or boolean.")
            await self.server.set_do(chan=key, value=bool(value))
        else:
            raise NotImplementedError(
                f"Setting value for type {self.kind} is not implemented."
            )

    async def get_with_uncertainty(
        self, keys: list[AI], acquisition_time: float = 1.0
    ) -> dict[AI, ufloat]:
        """
        Acquire AI data and return as ufloat values with uncertainty.

        This method collects data over the specified acquisition time and
        calculates mean ± standard error for each channel.

        Parameters
        ----------
        keys : list[AI]
            List of AI channel names to acquire
        acquisition_time : float, optional
            Acquisition time in seconds (default: 1.0)

        Returns
        -------
        dict[AI, ufloat]
            Dictionary mapping channel names to ufloat(mean, std_error)

        Raises
        ------
        TypeError
            If accessor is not for AI channels

        Examples
        --------
        >>> data = await server.ai.get_with_uncertainty(
        ...     keys=["Photodiode", "TEY signal"],
        ...     acquisition_time=2.0
        ... )
        >>> print(data["Photodiode"])  # 0.523 +/- 0.012
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
            data_array = np.array(chan_data["data"])

            # Calculate mean and standard error
            mean = np.nanmean(data_array)
            std_dev = np.nanstd(data_array, ddof=1)  # Sample std deviation
            std_error = std_dev / np.sqrt(len(data_array))

            uncertainty_data[chan_name] = ufloat(mean, std_error)

        return uncertainty_data

    async def table(self, keys: list[str]) -> TabularResponse:
        """
        Retrieve data for multiple keys and return in tabular format.

        Parameters
        ----------
        keys : list[str]
            List of keys to retrieve data for

        Returns
        -------
        TabularResponse
            Container with state and status DataFrames

        Examples
        --------
        >>> response = await server.motor.table(['Sample X', 'Sample Y'])
        >>> print(response.state)
           Sample X  Sample Y
        0      10.0       0.0
        >>> print(response.status)
              motor  position   goal  status
        0  Sample X      10.0   10.0       0
        1  Sample Y       0.0    0.0       0
        """
        records = []
        state = {}

        for key in keys:
            response = await self.__getitem__(key)

            if self.kind is AI:
                for chan_data in response["chans"]:
                    records.append(chan_data)
                    state[chan_data["chan"]] = chan_data["data"]

            elif self.kind is Motor:
                for motor_data in response["data"]:
                    state[motor_data["motor"]] = [motor_data["position"]]
                    records.append(
                        {
                            "motor": motor_data["motor"],
                            "position": motor_data["position"],
                            "goal": motor_data["goal"],
                            "status": motor_data["status"],
                            "time": motor_data["time"],
                        }
                    )

            elif self.kind is DIO:
                for chan, en, da in zip(
                    response["chans"], response["enabled"], response["data"]
                ):
                    state[chan] = da
                    records.append({"chan": chan, "enabled": en, "data": da})

        return TabularResponse(
            state=pd.DataFrame(state),
            status=pd.DataFrame(records),
        )

    async def table_with_uncertainty(
        self, keys: list[AI], acquisition_time: float = 1.0
    ) -> pd.DataFrame:
        """
        Retrieve AI data as DataFrame with mean and std columns.

        Parameters
        ----------
        keys : list[AI]
            List of AI channel names
        acquisition_time : float, optional
            Acquisition time in seconds (default: 1.0)

        Returns
        -------
        pd.DataFrame
            DataFrame with {channel}_mean and {channel}_std columns

        Examples
        --------
        >>> df = await server.ai.table_with_uncertainty(
        ...     keys=["Photodiode", "TEY signal"]
        ... )
        >>> print(df.columns)
        Index(['Photodiode_mean', 'Photodiode_std',
               'TEY signal_mean', 'TEY signal_std'])
        """
        if self.kind is not AI:
            raise TypeError("table_with_uncertainty only works with AI accessor")

        ufloat_data = await self.get_with_uncertainty(keys, acquisition_time)

        # Convert to DataFrame with _mean and _std columns
        data = {}
        for chan, uval in ufloat_data.items():
            data[f"{chan}_mean"] = [uval.nominal_value]
            data[f"{chan}_std"] = [uval.std_dev]

        return pd.DataFrame(data)

    async def from_df(self, df: pd.DataFrame) -> None:
        """
        Set multiple values from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns matching component names

        Raises
        ------
        PermissionError
            If accessor is read-only
        KeyError
            If any column name is not valid for this accessor type
        """
        if self.readonly:
            raise PermissionError("This accessor is read-only.")

        for key in df.columns:
            if key not in get_args(self.kind.__value__):
                raise KeyError(f"{key} is not a valid {self.kind}")

        for index, row in df.iterrows():
            for key in df.columns:
                await self.set(key, row[key])
