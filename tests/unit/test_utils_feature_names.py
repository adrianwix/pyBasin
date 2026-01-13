"""Tests for utility functions in pybasin.utils."""

from pybasin.utils import (
    get_feature_indices_by_base_name,
    parse_feature_name,
    validate_feature_names,
)


class TestParseFeatureName:
    """Tests for parse_feature_name function."""

    def test_valid_feature_name(self):
        result = parse_feature_name("state_0__variance")
        assert result == (0, "variance")

    def test_valid_feature_name_state_1(self):
        result = parse_feature_name("state_1__mean")
        assert result == (1, "mean")

    def test_valid_feature_name_with_nested_parts(self):
        result = parse_feature_name("state_0__autocorrelation_periodicity__output_strength")
        assert result == (0, "autocorrelation_periodicity__output_strength")

    def test_valid_feature_name_high_state_index(self):
        result = parse_feature_name("state_10__amplitude")
        assert result == (10, "amplitude")

    def test_invalid_feature_name_no_prefix(self):
        result = parse_feature_name("variance")
        assert result is None

    def test_invalid_feature_name_wrong_prefix(self):
        result = parse_feature_name("dim_0__variance")
        assert result is None

    def test_invalid_feature_name_missing_double_underscore(self):
        result = parse_feature_name("state_0_variance")
        assert result is None

    def test_invalid_feature_name_empty_string(self):
        result = parse_feature_name("")
        assert result is None

    def test_invalid_feature_name_only_prefix(self):
        result = parse_feature_name("state_0__")
        assert result is None


class TestValidateFeatureNames:
    """Tests for validate_feature_names function."""

    def test_all_valid_names(self):
        names = ["state_0__variance", "state_1__mean", "state_0__amplitude"]
        all_valid, invalid_names = validate_feature_names(names)
        assert all_valid is True
        assert invalid_names == []

    def test_some_invalid_names(self):
        names = ["state_0__variance", "invalid_name", "state_1__mean"]
        all_valid, invalid_names = validate_feature_names(names)
        assert all_valid is False
        assert invalid_names == ["invalid_name"]

    def test_all_invalid_names(self):
        names = ["bad_name", "another_bad", "wrong_format"]
        all_valid, invalid_names = validate_feature_names(names)
        assert all_valid is False
        assert len(invalid_names) == 3

    def test_empty_list(self):
        all_valid, invalid_names = validate_feature_names([])
        assert all_valid is True
        assert invalid_names == []

    def test_complex_valid_names(self):
        names = [
            "state_0__variance",
            "state_0__linear_trend__attr_slope",
            "state_1__autocorrelation_periodicity__output_strength",
        ]
        all_valid, invalid_names = validate_feature_names(names)
        assert all_valid is True
        assert invalid_names == []


class TestGetFeatureIndicesByBaseName:
    """Tests for get_feature_indices_by_base_name function."""

    def test_single_match(self):
        names = ["state_0__variance", "state_0__mean", "state_0__amplitude"]
        indices = get_feature_indices_by_base_name(names, "mean")
        assert indices == [1]

    def test_multiple_matches_across_states(self):
        names = [
            "state_0__variance",
            "state_1__variance",
            "state_0__mean",
            "state_1__mean",
        ]
        indices = get_feature_indices_by_base_name(names, "variance")
        assert indices == [0, 1]

    def test_nested_feature_name(self):
        names = [
            "state_0__autocorrelation_periodicity__output_strength",
            "state_0__autocorrelation_periodicity__output_period",
            "state_1__autocorrelation_periodicity__output_strength",
        ]
        indices = get_feature_indices_by_base_name(
            names, "autocorrelation_periodicity__output_strength"
        )
        assert indices == [0, 2]

    def test_no_matches(self):
        names = ["state_0__variance", "state_0__mean"]
        indices = get_feature_indices_by_base_name(names, "amplitude")
        assert indices == []

    def test_base_name_prefix_matching(self):
        names = [
            "state_0__autocorrelation_periodicity__output_strength",
            "state_0__autocorrelation_periodicity__output_period",
        ]
        indices = get_feature_indices_by_base_name(names, "autocorrelation_periodicity")
        assert indices == [0, 1]

    def test_empty_feature_names(self):
        indices = get_feature_indices_by_base_name([], "variance")
        assert indices == []

    def test_with_invalid_names_in_list(self):
        names = ["state_0__variance", "invalid_name", "state_1__variance"]
        indices = get_feature_indices_by_base_name(names, "variance")
        assert indices == [0, 2]

    def test_exact_match_preferred(self):
        names = [
            "state_0__mean",
            "state_0__mean_absolute",
        ]
        indices = get_feature_indices_by_base_name(names, "mean")
        assert indices == [0]
