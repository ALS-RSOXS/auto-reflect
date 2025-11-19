"""DataFrame validation utilities for scan plan creation"""

import re
from typing import List, Optional

import pandas as pd

from .types import ValidationError
from .types import motor as valid_motors

# ============================================================================
# Exposure Column Detection
# ============================================================================

EXPOSURE_PATTERNS = [
    r"^exposure$",
    r"^exp$",
    r"^count[_\s]?time",
    r"^integration[_\s]?time",
    r"^unnamed.*",  # Pandas unnamed column pattern
    r"^$",  # Empty string for final unnamed column
]


def find_exposure_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find exposure time column using pattern matching.

    Handles common variations:
    - "exposure", "exp"
    - "count_time", "count time"
    - "Unnamed: 2" (pandas default for unnamed columns)
    - "" (empty string column name)

    Parameters:
        df: Input DataFrame

    Returns:
        Column name if found, None otherwise
    """
    for col in df.columns:
        col_str = str(col).lower().strip()
        for pattern in EXPOSURE_PATTERNS:
            if re.match(pattern, col_str, re.IGNORECASE):
                return col
    return None


# ============================================================================
# Motor Column Validation
# ============================================================================


def validate_motor_columns(df: pd.DataFrame) -> List[str]:
    """
    Validate that DataFrame columns match known motor names.

    Parameters:
        df: Input DataFrame

    Returns:
        List of valid motor column names

    Raises:
        ValidationError: If invalid columns found or no motor columns present
    """
    motor_cols = []
    exposure_col = find_exposure_column(df)

    for col in df.columns:
        # Skip exposure column
        if col == exposure_col:
            continue

        # Check if column is a valid motor name
        if col not in valid_motors:
            # Provide helpful error message
            similar_motors = [m for m in valid_motors if col.lower() in m.lower()]
            error_msg = f"Column '{col}' is not a valid motor name."
            if similar_motors:
                error_msg += f"\n  Did you mean one of: {similar_motors[:3]}?"
            else:
                error_msg += f"\n  Valid motors include: {list(valid_motors)[:5]}..."
            raise ValidationError(error_msg)

        motor_cols.append(col)

    if not motor_cols:
        raise ValidationError(
            "DataFrame must contain at least one motor column.\n"
            f"Valid motor names: {list(valid_motors)[:10]}..."
        )

    return motor_cols


# ============================================================================
# DataFrame Validation
# ============================================================================


def validate_scan_dataframe(df: pd.DataFrame) -> tuple[List[str], Optional[str]]:
    """
    Validate complete scan DataFrame.

    Parameters:
        df: Input DataFrame with motor columns and optional exposure column

    Returns:
        Tuple of (motor_column_names, exposure_column_name)

    Raises:
        ValidationError: If validation fails
    """
    if df.empty:
        raise ValidationError("DataFrame is empty")

    # Find exposure column
    exposure_col = find_exposure_column(df)

    # Validate motor columns
    motor_cols = validate_motor_columns(df)

    # Validate exposure values if column exists
    if exposure_col is not None:
        exposure_values = df[exposure_col]

        # Check for NaN values in exposure
        if exposure_values.isna().any():
            nan_indices = df[exposure_values.isna()].index.tolist()
            raise ValidationError(
                f"NaN values found in exposure column '{exposure_col}' at rows: {nan_indices}"
            )

        # Check for invalid values
        if (exposure_values <= 0).any():
            invalid_indices = df[exposure_values <= 0].index.tolist()
            raise ValidationError(
                f"Invalid exposure times (must be > 0) at rows: {invalid_indices}"
            )

    # Check for NaN values in motor columns
    for col in motor_cols:
        if df[col].isna().any():
            nan_indices = df[df[col].isna()].index.tolist()
            raise ValidationError(
                f"NaN values found in motor column '{col}' at rows: {nan_indices}"
            )

    return motor_cols, exposure_col
