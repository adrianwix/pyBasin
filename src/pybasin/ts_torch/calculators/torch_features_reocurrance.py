# pyright: basic
import torch
from torch import Tensor

# =============================================================================
# REOCCURRENCE FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def percentage_of_reoccurring_datapoints_to_all_datapoints(x: Tensor) -> Tensor:
    """Percentage of unique values that appear more than once (fully vectorized)."""
    n, batch_size, n_states = x.shape

    x_flat = x.reshape(n, -1).T  # (B*S, N)

    sorted_x, _ = x_flat.sort(dim=1)

    is_dup = sorted_x[:, 1:] == sorted_x[:, :-1]  # (B*S, N-1)

    is_dup_prev = torch.cat(
        [torch.zeros(batch_size * n_states, 1, dtype=torch.bool, device=x.device), is_dup],
        dim=1,
    )
    is_dup_next = torch.cat(
        [is_dup, torch.zeros(batch_size * n_states, 1, dtype=torch.bool, device=x.device)],
        dim=1,
    )
    is_part_of_dup_group = is_dup_prev | is_dup_next  # (B*S, N)

    is_first_in_group = (~is_dup_prev) & is_part_of_dup_group
    num_reoccurring_unique = is_first_in_group.sum(dim=1).float()

    is_unique_start = torch.cat(
        [
            torch.ones(batch_size * n_states, 1, dtype=torch.bool, device=x.device),
            sorted_x[:, 1:] != sorted_x[:, :-1],
        ],
        dim=1,
    )
    num_unique = is_unique_start.sum(dim=1).float()

    result = num_reoccurring_unique / (num_unique + 1e-10)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def percentage_of_reoccurring_values_to_all_values(x: Tensor) -> Tensor:
    """Percentage of datapoints that are reoccurring (optimized)."""
    n, batch_size, n_states = x.shape

    if n <= 1:
        return torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Sort and find adjacent duplicates
    sorted_x, _ = x_flat.sort(dim=0)
    is_dup = sorted_x[1:] == sorted_x[:-1]  # (N-1, B*S)

    # Mark all positions that are part of a duplicate group
    # Position i is reoccurring if sorted[i] == sorted[i-1] OR sorted[i] == sorted[i+1]
    is_dup_prev = torch.cat(
        [torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device), is_dup], dim=0
    )
    is_dup_next = torch.cat(
        [is_dup, torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device)], dim=0
    )
    is_reoccurring = is_dup_prev | is_dup_next  # (N, B*S)

    # Count reoccurring values
    result = is_reoccurring.float().sum(dim=0) / n  # (B*S,)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def sum_of_reoccurring_data_points(x: Tensor) -> Tensor:
    """Sum of values that appear more than once (optimized)."""
    n, batch_size, n_states = x.shape

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Sort and find duplicates
    sorted_x, _ = x_flat.sort(dim=0)
    is_dup = sorted_x[1:] == sorted_x[:-1]  # (N-1, B*S)

    # Mark all values that are duplicated (appear more than once)
    is_dup_prev = torch.cat(
        [torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device), is_dup], dim=0
    )
    is_dup_next = torch.cat(
        [is_dup, torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device)], dim=0
    )
    is_reoccurring = is_dup_prev | is_dup_next  # (N, B*S)

    # Sum reoccurring values
    result = (sorted_x * is_reoccurring.float()).sum(dim=0)  # (B*S,)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def sum_of_reoccurring_values(x: Tensor) -> Tensor:
    """Sum of unique values that appear more than once (optimized)."""
    n, batch_size, n_states = x.shape

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Sort and find duplicate boundaries
    sorted_x, _ = x_flat.sort(dim=0)
    is_dup = sorted_x[1:] == sorted_x[:-1]  # (N-1, B*S)

    # Find first occurrence of each reoccurring value
    # A value is the "first of reoccurring" if it's followed by a duplicate but not preceded by one
    is_dup_prev = torch.cat(
        [torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device), is_dup], dim=0
    )
    is_dup_next = torch.cat(
        [is_dup, torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device)], dim=0
    )

    # First of a duplicate run: has duplicate after but not before
    is_first_of_run = is_dup_next & ~is_dup_prev  # (N, B*S)

    # Sum unique reoccurring values (just the first of each run)
    result = (sorted_x * is_first_of_run.float()).sum(dim=0)  # (B*S,)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def ratio_value_number_to_time_series_length(x: Tensor) -> Tensor:
    """Ratio of unique values to length (optimized)."""
    n, batch_size, n_states = x.shape

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Sort and count duplicates
    sorted_x, _ = x_flat.sort(dim=0)
    is_dup = (sorted_x[1:] == sorted_x[:-1]).float()  # (N-1, B*S)

    # Number of unique = N - number of duplicates
    n_dups = is_dup.sum(dim=0)  # (B*S,)
    n_unique = n - n_dups

    result = n_unique / n
    return result.reshape(batch_size, n_states)
