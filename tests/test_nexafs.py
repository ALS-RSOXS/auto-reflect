"""
Tests for NEXAFS functionality.

Tests NEXAFS absorption calculation and error propagation.
"""

import numpy as np
import pandas as pd

from api_dev.nexafs import (
    calculate_nexafs,
    normalize_to_edge_jump,
)


class TestCalculateNexafs:
    """Tests for calculate_nexafs function."""

    def test_basic_calculation(self):
        """Test basic NEXAFS calculation."""
        # Create mock scan data
        scan_data = pd.DataFrame(
            {
                "Beamline Energy_position": [280, 285, 290, 295, 300],
                "TEY signal_mean": [0.5, 0.6, 0.7, 0.8, 0.9],
                "TEY signal_std": [0.01, 0.01, 0.01, 0.01, 0.01],
                "AI 3 Izero_mean": [1.0, 1.0, 1.0, 1.0, 1.0],
                "AI 3 Izero_std": [0.02, 0.02, 0.02, 0.02, 0.02],
            }
        )

        result = calculate_nexafs(scan_data)

        # Check output columns
        assert "energy" in result.columns
        assert "transmission_mean" in result.columns
        assert "transmission_std" in result.columns
        assert "absorption_mean" in result.columns
        assert "absorption_std" in result.columns

        # Check transmission = signal / i0
        expected_transmission = (
            scan_data["TEY signal_mean"] / scan_data["AI 3 Izero_mean"]
        )
        np.testing.assert_array_almost_equal(
            result["transmission_mean"], expected_transmission
        )

        # Check absorption = -ln(transmission)
        expected_absorption = -np.log(expected_transmission)
        np.testing.assert_array_almost_equal(
            result["absorption_mean"], expected_absorption
        )

    def test_custom_channel_names(self):
        """Test with custom channel names."""
        scan_data = pd.DataFrame(
            {
                "Beamline Energy_position": [280, 285],
                "Photodiode_mean": [0.5, 0.6],
                "Photodiode_std": [0.01, 0.01],
                "AI 1_mean": [1.0, 1.0],
                "AI 1_std": [0.02, 0.02],
            }
        )

        result = calculate_nexafs(
            scan_data, signal_channel="Photodiode", normalization_channel="AI 1"
        )

        assert len(result) == 2
        assert "Photodiode_mean" in result.columns
        assert "AI 1_mean" in result.columns

    def test_invalid_transmission_filtered(self):
        """Test that invalid transmission values are filtered."""
        scan_data = pd.DataFrame(
            {
                "Beamline Energy_position": [280, 285, 290],
                "TEY signal_mean": [0.5, 0.0, -0.1],  # Invalid values
                "TEY signal_std": [0.01, 0.01, 0.01],
                "AI 3 Izero_mean": [1.0, 1.0, 1.0],
                "AI 3 Izero_std": [0.02, 0.02, 0.02],
            }
        )

        result = calculate_nexafs(scan_data)

        # Should only keep the first point (transmission = 0.5)
        assert len(result) == 1
        assert result["energy"].iloc[0] == 280

    def test_error_propagation(self):
        """Test that uncertainties are properly propagated."""
        scan_data = pd.DataFrame(
            {
                "Beamline Energy_position": [280],
                "TEY signal_mean": [0.5],
                "TEY signal_std": [0.05],  # 10% uncertainty
                "AI 3 Izero_mean": [1.0],
                "AI 3 Izero_std": [0.1],  # 10% uncertainty
            }
        )

        result = calculate_nexafs(scan_data)

        # Transmission uncertainty should be propagated
        assert result["transmission_std"].iloc[0] > 0

        # Absorption uncertainty should be propagated
        assert result["absorption_std"].iloc[0] > 0


class TestNormalizeToEdgeJump:
    """Tests for normalize_to_edge_jump function."""

    def test_basic_normalization(self):
        """Test basic edge jump normalization."""
        # Create mock NEXAFS data with clear edge
        energies = np.linspace(280, 300, 100)
        absorption = np.ones(100)
        absorption[50:] = 2.0  # Edge jump at E=290

        nexafs_data = pd.DataFrame(
            {
                "energy": energies,
                "absorption_mean": absorption,
                "absorption_std": np.full(100, 0.01),
            }
        )

        result = normalize_to_edge_jump(
            nexafs_data, pre_edge_range=(280, 285), post_edge_range=(295, 300)
        )

        # Check normalization columns exist
        assert "absorption_normalized_mean" in result.columns
        assert "absorption_normalized_std" in result.columns

        # Pre-edge should be ~0, post-edge should be ~1
        pre_norm = result[result["energy"] < 285]["absorption_normalized_mean"].mean()
        post_norm = result[result["energy"] > 295]["absorption_normalized_mean"].mean()

        assert abs(pre_norm) < 0.1  # Close to 0
        assert abs(post_norm - 1.0) < 0.1  # Close to 1

    def test_uncertainty_propagation(self):
        """Test that normalization propagates uncertainties."""
        energies = np.linspace(280, 300, 50)
        nexafs_data = pd.DataFrame(
            {
                "energy": energies,
                "absorption_mean": np.linspace(1.0, 2.0, 50),
                "absorption_std": np.full(50, 0.05),
            }
        )

        result = normalize_to_edge_jump(
            nexafs_data, pre_edge_range=(280, 285), post_edge_range=(295, 300)
        )

        # Uncertainties should be propagated
        assert np.all(result["absorption_normalized_std"] > 0)
