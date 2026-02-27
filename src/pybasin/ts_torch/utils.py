from torch import Tensor

from pybasin.feature_extractors.utils import format_feature_name
from pybasin.ts_torch.settings import (
    ALL_FEATURE_FUNCTIONS,
    TORCH_COMPREHENSIVE_FC_PARAMETERS,
    TORCH_CUSTOM_FC_PARAMETERS,
    FCParameters,
)


def get_feature_names_from_config(
    fc_parameters: FCParameters | None = None,
    include_custom: bool = False,
) -> list[str]:
    """Get list of feature names from configuration.

    Parses the feature configuration dictionary to generate a list of all feature names
    that will be computed, including parameterized variants.

    :param fc_parameters: Feature configuration dictionary mapping feature names to parameter
        lists. If None, uses TORCH_COMPREHENSIVE_FC_PARAMETERS as default. Each entry can be:
        - None: feature with no parameters
        - list of dicts: feature with multiple parameter combinations
    :param include_custom: Whether to include custom features (delta, log_delta) in the output.
        Default is False.
    :return: List of feature name strings. For parameterized features, names include parameters
        in the format "feature_name__param1_value1__param2_value2". The order matches the order
        features will be computed.
    """
    if fc_parameters is None:
        fc_parameters = TORCH_COMPREHENSIVE_FC_PARAMETERS

    names: list[str] = []

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue
        if param_list is None:
            names.append(feature_name)
        else:
            for params in param_list:
                names.append(format_feature_name(feature_name, params))

    if include_custom:
        for feature_name in TORCH_CUSTOM_FC_PARAMETERS:
            if feature_name in ALL_FEATURE_FUNCTIONS:
                names.append(feature_name)

    return names


def extract_features_from_config(
    x: Tensor,
    fc_parameters: FCParameters | None = None,
    include_custom: bool = False,
) -> dict[str, Tensor]:
    """Extract features from tensor using configuration.

    Computes time series features according to the provided configuration, applying
    each feature function (with its parameters) to the input time series data.

    :param x: Input tensor of shape (N, B, S) where:
        - N: number of time points in the time series
        - B: batch size (number of different initial conditions/samples)
        - S: number of state variables
    :param fc_parameters: Feature configuration dictionary mapping feature names to parameter
        lists. If None, uses TORCH_COMPREHENSIVE_FC_PARAMETERS as default.
    :param include_custom: Whether to include custom features (delta, log_delta). Default is False.
    :return: Dictionary mapping feature names to result tensors of shape (B, S). Each key is a
        feature name (possibly with parameters like "feature__param_value"), and each value is
        a tensor containing that feature computed for all batches and states.
    """
    if fc_parameters is None:
        fc_parameters = TORCH_COMPREHENSIVE_FC_PARAMETERS

    results: dict[str, Tensor] = {}

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue
        func = ALL_FEATURE_FUNCTIONS[feature_name]

        if param_list is None:
            results[feature_name] = func(x)
        else:
            for params in param_list:
                name = format_feature_name(feature_name, params)
                results[name] = func(x, **params)

    if include_custom:
        for feature_name in TORCH_CUSTOM_FC_PARAMETERS:
            if feature_name in ALL_FEATURE_FUNCTIONS:
                results[feature_name] = ALL_FEATURE_FUNCTIONS[feature_name](x)

    return results
