"""Tests to verify all JAX_COMPREHENSIVE_FC_PARAMETERS features are JIT compilable."""

import jax
import jax.numpy as jnp
import pytest

from pybasin.feature_extractors.jax_feature_calculators import (
    ALL_FEATURE_FUNCTIONS,
    JAX_COMPREHENSIVE_FC_PARAMETERS,
)


def get_test_cases():
    """Generate test cases - ONE config per feature for speed (73 tests, not 789)."""
    test_cases = []
    for feature_name, param_value in JAX_COMPREHENSIVE_FC_PARAMETERS.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue
        if param_value is None:
            test_cases.append((feature_name, None, feature_name))
        else:
            # Only test the first parameter configuration for speed
            first_params = param_value[0]
            case_id = f"{feature_name}_{first_params}"
            test_cases.append((feature_name, first_params, case_id))
    return test_cases


TEST_CASES = get_test_cases()


@pytest.fixture
def sample_data():
    """Create sample time series data for testing (N, B, S)."""
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, shape=(100, 10, 3))


@pytest.mark.parametrize(
    "feature_name,params,case_id",
    TEST_CASES,
    ids=[tc[2] for tc in TEST_CASES],
)
def test_jit_compiles(
    feature_name: str, params: dict | None, case_id: str, sample_data: jnp.ndarray
):
    """Test that the feature function can be JIT compiled and executed."""
    func = ALL_FEATURE_FUNCTIONS[feature_name]

    if params:
        jitted_fn = jax.jit(lambda x, p=params: func(x, **p))
    else:
        jitted_fn = jax.jit(func)

    result = jitted_fn(sample_data)

    assert result is not None
    assert jnp.isfinite(result).all() or jnp.isnan(result).all(), (
        f"{case_id} produced invalid values"
    )
