from typing import Any
from unittest.mock import MagicMock, patch

from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.study_params import SweepStudyParams


def test_adaptive_study_passes_sampler_correctly() -> None:
    """Test that ASBasinStabilityEstimator passes different samplers to BSE."""

    # Create mock samplers (disable __len__ to prevent numpy array issues)
    sampler1 = MagicMock(name="sampler1")
    sampler2 = MagicMock(name="sampler2")
    sampler3 = MagicMock(name="sampler3")
    sampler1.__len__ = None
    sampler2.__len__ = None
    sampler3.__len__ = None

    # Use a Python list for object parameter values
    sampler_values = [sampler1, sampler2, sampler3]

    study_params = SweepStudyParams(
        name="sampler",
        values=sampler_values,
    )

    # Mock all dependencies
    mock_ode = MagicMock()
    mock_solver = MagicMock()
    mock_feature_extractor = MagicMock()
    mock_predictor = MagicMock()

    # Track which samplers are passed to BasinStabilityEstimator
    captured_samplers: list[Any] = []

    def capture_bse_init(*args: Any, **kwargs: Any) -> MagicMock:
        """Capture the sampler passed to BSE."""
        captured_samplers.append(kwargs.get("sampler"))
        mock_bse = MagicMock()
        mock_bse.estimate_bs.return_value = {"attractor1": 0.5, "attractor2": 0.5}
        mock_bse.get_errors.return_value = {}
        mock_bse.solution = None
        mock_bse.y0 = None
        mock_bse.n = 100
        return mock_bse

    with patch(
        "pybasin.as_basin_stability_estimator.BasinStabilityEstimator", side_effect=capture_bse_init
    ):
        bse = ASBasinStabilityEstimator(
            n=100,
            ode_system=mock_ode,
            sampler=MagicMock(),  # Default sampler (won't be used)
            solver=mock_solver,
            feature_extractor=mock_feature_extractor,
            cluster_classifier=mock_predictor,
            study_params=study_params,
            save_to=None,
        )

        bse.estimate_as_bs()

    # Verify the correct samplers were passed
    assert len(captured_samplers) == 3
    assert captured_samplers[0] is sampler1
    assert captured_samplers[1] is sampler2
    assert captured_samplers[2] is sampler3
    print("✓ All samplers were correctly passed to BasinStabilityEstimator!")
