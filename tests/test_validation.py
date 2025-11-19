"""
Tests for validation module.

Tests DataFrame validation, exposure column detection, and motor column validation.
"""

import numpy as np
import pandas as pd
import pytest

from api_dev.types import ValidationError
from api_dev.validation import (
    find_exposure_column,
    validate_motor_columns,
    validate_scan_dataframe,
)


class TestFindExposureColumn:
    """Tests for find_exposure_column function."""

    def test_exposure_column(self):
        """Test detection of 'exposure' column."""
        df = pd.DataFrame({"Sample X": [1, 2], "exposure": [1.0, 1.5]})
        assert find_exposure_column(df) == "exposure"

    def test_exp_column(self):
        """Test detection of 'exp' column."""
        df = pd.DataFrame({"Sample X": [1, 2], "exp": [1.0, 1.5]})
        assert find_exposure_column(df) == "exp"

    def test_count_time_column(self):
        """Test detection of 'count_time' column."""
        df = pd.DataFrame({"Sample X": [1, 2], "count_time": [1.0, 1.5]})
        assert find_exposure_column(df) == "count_time"

    def test_unnamed_column(self):
        """Test detection of unnamed columns."""
        df = pd.DataFrame({"Sample X": [1, 2], "Unnamed: 2": [1.0, 1.5]})
        assert find_exposure_column(df) == "Unnamed: 2"

    def test_empty_column_name(self):
        """Test detection of empty string column."""
        df = pd.DataFrame({"Sample X": [1, 2], "": [1.0, 1.5]})
        assert find_exposure_column(df) == ""

    def test_case_insensitive(self):
        """Test case-insensitive detection."""
        df = pd.DataFrame({"Sample X": [1, 2], "EXPOSURE": [1.0, 1.5]})
        assert find_exposure_column(df) == "EXPOSURE"

    def test_no_exposure_column(self):
        """Test when no exposure column exists."""
        df = pd.DataFrame({"Sample X": [1, 2], "Sample Y": [0, 0]})
        assert find_exposure_column(df) is None


class TestValidateMotorColumns:
    """Tests for validate_motor_columns function."""

    def test_valid_motor_columns(self):
        """Test with valid motor columns."""
        df = pd.DataFrame(
            {"Sample X": [1, 2], "Sample Y": [0, 0], "exposure": [1.0, 1.5]}
        )
        cols = validate_motor_columns(df)
        assert "Sample X" in cols
        assert "Sample Y" in cols
        assert "exposure" not in cols

    def test_single_motor_column(self):
        """Test with single motor column."""
        df = pd.DataFrame({"Sample X": [1, 2], "exposure": [1.0, 1.5]})
        cols = validate_motor_columns(df)
        assert cols == ["Sample X"]

    def test_invalid_motor_name(self):
        """Test with invalid motor name."""
        df = pd.DataFrame({"Invalid Motor": [1, 2], "exposure": [1.0, 1.5]})
        with pytest.raises(ValidationError, match="Invalid Motor"):
            validate_motor_columns(df)

    def test_no_motor_columns(self):
        """Test with no motor columns."""
        df = pd.DataFrame({"exposure": [1.0, 1.5]})
        with pytest.raises(ValidationError, match="at least one motor"):
            validate_motor_columns(df)

    def test_mixed_valid_invalid(self):
        """Test with mix of valid and invalid columns."""
        df = pd.DataFrame(
            {"Sample X": [1, 2], "Invalid Motor": [0, 0], "exposure": [1.0, 1.5]}
        )
        with pytest.raises(ValidationError, match="Invalid Motor"):
            validate_motor_columns(df)


class TestValidateScanDataFrame:
    """Tests for validate_scan_dataframe function."""

    def test_valid_dataframe(self):
        """Test with valid scan DataFrame."""
        df = pd.DataFrame(
            {
                "Sample X": [10.0, 10.5, 11.0],
                "Sample Y": [0.0, 0.0, 0.0],
                "exposure": [1.0, 1.5, 2.0],
            }
        )
        motor_cols, exposure_col = validate_scan_dataframe(df)
        assert set(motor_cols) == {"Sample X", "Sample Y"}
        assert exposure_col == "exposure"

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            validate_scan_dataframe(df)

    def test_negative_exposure(self):
        """Test with negative exposure time."""
        df = pd.DataFrame(
            {
                "Sample X": [1, 2],
                "exposure": [1.0, -0.5],  # Negative!
            }
        )
        with pytest.raises(ValidationError, match="Invalid exposure"):
            validate_scan_dataframe(df)

    def test_nan_in_motor_column(self):
        """Test with NaN in motor column."""
        df = pd.DataFrame({"Sample X": [1.0, np.nan, 3.0], "exposure": [1.0, 1.0, 1.0]})
        with pytest.raises(ValidationError, match="NaN"):
            validate_scan_dataframe(df)

    def test_nan_in_exposure_column(self):
        """Test with NaN in exposure column."""
        df = pd.DataFrame({"Sample X": [1.0, 2.0, 3.0], "exposure": [1.0, np.nan, 1.0]})
        with pytest.raises(ValidationError, match="NaN"):
            validate_scan_dataframe(df)

    def test_no_exposure_column(self):
        """Test with no exposure column (should be valid - exposure is optional)."""
        df = pd.DataFrame({"Sample X": [1, 2], "Sample Y": [0, 0]})
        # Should succeed with None for exposure column
        motor_cols, exposure_col = validate_scan_dataframe(df)
        assert set(motor_cols) == {"Sample X", "Sample Y"}
        assert exposure_col is None

    def test_multiple_motors(self):
        """Test with multiple motor columns."""
        df = pd.DataFrame(
            {
                "Sample X": [1.0, 2.0],
                "Sample Y": [0.0, 0.5],
                "Beamline Energy": [285.0, 290.0],
                "exposure": [1.0, 1.5],
            }
        )
        motor_cols, exposure_col = validate_scan_dataframe(df)
        assert len(motor_cols) == 3
        assert "Sample X" in motor_cols
        assert "Sample Y" in motor_cols
        assert "Beamline Energy" in motor_cols
