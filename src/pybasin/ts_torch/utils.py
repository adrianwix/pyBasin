from torch import Tensor

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

    Args:
        fc_parameters: Feature configuration (None for defaults)
        include_custom: Include custom features

    Returns:
        List of feature names
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
                names.append(_format_feature_name(feature_name, params))

    if include_custom:
        for feature_name in TORCH_CUSTOM_FC_PARAMETERS:
            if feature_name in ALL_FEATURE_FUNCTIONS:
                names.append(feature_name)

    return names


def _format_feature_name(feature_name: str, params: dict[str, object] | None) -> str:
    """Format feature name with parameters."""
    if params is None:
        return feature_name
    param_str = "__".join(f"{k}_{v}" for k, v in sorted(params.items()))
    return f"{feature_name}__{param_str}"


def extract_features_from_config(
    x: Tensor,
    fc_parameters: FCParameters | None = None,
    include_custom: bool = False,
) -> dict[str, Tensor]:
    """Extract features from tensor using configuration.

    Args:
        x: Input tensor of shape (N, B, S)
        fc_parameters: Feature configuration (None for defaults)
        include_custom: Include custom features (delta, log_delta)

    Returns:
        Dictionary mapping feature names to result tensors of shape (B, S)
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
                name = _format_feature_name(feature_name, params)
                results[name] = func(x, **params)

    if include_custom:
        for feature_name in TORCH_CUSTOM_FC_PARAMETERS:
            if feature_name in ALL_FEATURE_FUNCTIONS:
                results[feature_name] = ALL_FEATURE_FUNCTIONS[feature_name](x)

    return results
