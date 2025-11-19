"""
Pytest configuration and shared fixtures for api_dev tests.
"""

import pytest


@pytest.fixture
def sample_scan_dataframe():
    """Fixture providing a basic valid scan DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        {
            "Sample X": [10.0, 10.5, 11.0, 11.5, 12.0],
            "Sample Y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "exposure": [1.0, 1.0, 1.5, 1.5, 2.0],
        }
    )


@pytest.fixture
def sample_scan_results():
    """Fixture providing mock scan results with uncertainties."""
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "index": [0, 1, 2, 3, 4],
            "Sample X_position": [10.0, 10.5, 11.0, 11.5, 12.0],
            "Sample Y_position": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Photodiode_mean": [0.45, 0.52, 0.58, 0.63, 0.68],
            "Photodiode_std": [0.01, 0.01, 0.01, 0.01, 0.01],
            "TEY signal_mean": [0.35, 0.42, 0.48, 0.53, 0.58],
            "TEY signal_std": [0.02, 0.02, 0.02, 0.02, 0.02],
            "exposure": [1.0, 1.0, 1.5, 1.5, 2.0],
            "timestamp": np.linspace(1000, 1020, 5),
        }
    )


@pytest.fixture
def nexafs_scan_dataframe():
    """Fixture providing NEXAFS energy scan DataFrame."""
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {"Beamline Energy": np.linspace(280, 320, 50), "exposure": np.ones(50)}
    )


@pytest.fixture
def nexafs_results():
    """Fixture providing mock NEXAFS results with uncertainties."""
    import numpy as np
    import pandas as pd

    energies = np.linspace(280, 320, 50)

    # Simulate absorption edge
    transmission = np.ones(50)
    transmission[energies > 285] = 0.7  # Edge jump
    transmission[energies > 295] = 0.5

    return pd.DataFrame(
        {
            "Beamline Energy_position": energies,
            "TEY signal_mean": transmission * 0.8,
            "TEY signal_std": transmission * 0.01,
            "AI 3 Izero_mean": np.ones(50),
            "AI 3 Izero_std": np.ones(50) * 0.02,
        }
    )
