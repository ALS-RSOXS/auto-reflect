"""
Tests for utility functions.

Tests scan generation utilities and data analysis helpers.
"""

import numpy as np
import pandas as pd
import pytest

from api_dev.utils import (
    calculate_center_of_mass,
    create_energy_scan,
    create_grid_scan,
    create_line_scan,
    find_peak_position,
)


class TestCreateGridScan:
    """Tests for create_grid_scan function."""

    def test_basic_grid(self):
        """Test basic grid scan creation."""
        grid = create_grid_scan(
            x_range=(10, 12, 3), y_range=(0, 2, 3), exposure_time=1.0
        )

        # Should have 3x3 = 9 points
        assert len(grid) == 9

        # Check columns
        assert "Sample X" in grid.columns
        assert "Sample Y" in grid.columns
        assert "exposure" in grid.columns

        # Check X range
        assert grid["Sample X"].min() == 10.0
        assert grid["Sample X"].max() == 12.0

        # Check Y range
        assert grid["Sample Y"].min() == 0.0
        assert grid["Sample Y"].max() == 2.0

        # Check exposure
        assert (grid["exposure"] == 1.0).all()

    def test_custom_motor_names(self):
        """Test with custom motor names."""
        grid = create_grid_scan(
            x_range=(0, 10, 5), y_range=(0, 5, 3), x_motor="Motor A", y_motor="Motor B"
        )

        assert "Motor A" in grid.columns
        assert "Motor B" in grid.columns

    def test_single_point_grid(self):
        """Test with single point (1x1 grid)."""
        grid = create_grid_scan(x_range=(10, 10, 1), y_range=(5, 5, 1))

        assert len(grid) == 1
        assert grid["Sample X"].iloc[0] == 10.0
        assert grid["Sample Y"].iloc[0] == 5.0


class TestCreateLineScan:
    """Tests for create_line_scan function."""

    def test_basic_line_scan(self):
        """Test basic line scan creation."""
        scan = create_line_scan(
            motor="Sample X", start=10, stop=15, num_points=6, exposure_time=1.5
        )

        assert len(scan) == 6
        assert "Sample X" in scan.columns
        assert "exposure" in scan.columns

        # Check positions
        expected = np.linspace(10, 15, 6)
        np.testing.assert_array_almost_equal(scan["Sample X"], expected)

        # Check exposure
        assert (scan["exposure"] == 1.5).all()

    def test_single_point_line(self):
        """Test with single point."""
        scan = create_line_scan(motor="Sample Y", start=5, stop=5, num_points=1)

        assert len(scan) == 1
        assert scan["Sample Y"].iloc[0] == 5.0


class TestCreateEnergyScan:
    """Tests for create_energy_scan function."""

    def test_constant_exposure(self):
        """Test with constant exposure time."""
        energies = np.linspace(280, 320, 100)
        scan = create_energy_scan(energies, exposure_time=1.0)

        assert len(scan) == 100
        assert "Beamline Energy" in scan.columns
        assert "exposure" in scan.columns

        np.testing.assert_array_almost_equal(scan["Beamline Energy"], energies)
        assert (scan["exposure"] == 1.0).all()

    def test_variable_exposure(self):
        """Test with variable exposure times."""
        energies = np.linspace(280, 320, 10)
        exposure = np.linspace(0.5, 2.0, 10)

        scan = create_energy_scan(energies, exposure_time=exposure)

        np.testing.assert_array_almost_equal(scan["exposure"], exposure)

    def test_custom_motor_name(self):
        """Test with custom energy motor name."""
        energies = np.array([280, 285, 290])
        scan = create_energy_scan(energies, energy_motor="Monochromator Energy")

        assert "Monochromator Energy" in scan.columns


class TestFindPeakPosition:
    """Tests for find_peak_position function."""

    def test_basic_peak_finding(self):
        """Test basic peak finding."""
        # Create data with peak at x=10
        scan_data = pd.DataFrame(
            {
                "Sample X_position": [8, 9, 10, 11, 12],
                "Photodiode_mean": [0.2, 0.5, 1.0, 0.5, 0.2],
            }
        )

        peak_x, peak_value = find_peak_position(
            scan_data, motor_col="Sample X_position", signal_col="Photodiode_mean"
        )

        assert peak_x == 10
        assert peak_value == 1.0

    def test_multiple_peaks(self):
        """Test with multiple peaks (finds maximum)."""
        scan_data = pd.DataFrame(
            {
                "Sample X_position": [0, 1, 2, 3, 4],
                "Photodiode_mean": [0.5, 0.3, 0.8, 0.3, 0.6],
            }
        )

        peak_x, peak_value = find_peak_position(
            scan_data, motor_col="Sample X_position", signal_col="Photodiode_mean"
        )

        assert peak_x == 2
        assert peak_value == 0.8

    def test_auto_append_mean(self):
        """Test automatic _mean suffix addition."""
        scan_data = pd.DataFrame(
            {"Sample X_position": [1, 2, 3], "Photodiode_mean": [0.1, 0.5, 0.2]}
        )

        # Should automatically add _mean suffix
        peak_x, peak_value = find_peak_position(
            scan_data,
            motor_col="Sample X_position",
            signal_col="Photodiode",  # No _mean suffix
            use_mean=True,
        )

        assert peak_value == 0.5


class TestCalculateCenterOfMass:
    """Tests for calculate_center_of_mass function."""

    def test_symmetric_distribution(self):
        """Test with symmetric distribution."""
        # Symmetric Gaussian-like distribution centered at 10
        scan_data = pd.DataFrame(
            {
                "Sample X_position": [8, 9, 10, 11, 12],
                "Photodiode_mean": [0.2, 0.6, 1.0, 0.6, 0.2],
            }
        )

        com = calculate_center_of_mass(
            scan_data, motor_col="Sample X_position", signal_col="Photodiode_mean"
        )

        # Should be very close to 10
        assert abs(com - 10.0) < 0.1

    def test_asymmetric_distribution(self):
        """Test with asymmetric distribution."""
        scan_data = pd.DataFrame(
            {
                "Sample X_position": [1, 2, 3, 4, 5],
                "Photodiode_mean": [0.1, 0.2, 0.5, 0.8, 0.3],
            }
        )

        com = calculate_center_of_mass(
            scan_data, motor_col="Sample X_position", signal_col="Photodiode_mean"
        )

        # COM should be between 3 and 4 (skewed toward peak)
        assert 3.0 < com < 4.5

    def test_zero_signal_error(self):
        """Test that zero signal raises error."""
        scan_data = pd.DataFrame(
            {"Sample X_position": [1, 2, 3], "Photodiode_mean": [0.0, 0.0, 0.0]}
        )

        with pytest.raises(ValueError, match="zero"):
            calculate_center_of_mass(
                scan_data, motor_col="Sample X_position", signal_col="Photodiode_mean"
            )

    def test_negative_signal_handled(self):
        """Test that negative signals are shifted to positive."""
        scan_data = pd.DataFrame(
            {"Sample X_position": [1, 2, 3], "Photodiode_mean": [-0.5, 0.0, 0.5]}
        )

        # Should shift all values positive and calculate
        com = calculate_center_of_mass(
            scan_data, motor_col="Sample X_position", signal_col="Photodiode_mean"
        )

        # COM should be weighted toward higher value (position 3)
        assert 2.0 < com < 3.0
