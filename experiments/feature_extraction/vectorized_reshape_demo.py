"""Demo: vectorized reshape from (N, B, S) tensor to tsfresh wide DataFrame.

Shows how transpose + reshape replaces a triple-nested Python loop.
"""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def print_array_aligned(label: str, arr: np.ndarray, indent: int = 2) -> None:
    """Print array with proper indentation for multi-line output."""
    arr_str = np.array2string(arr, separator=" ")
    lines = arr_str.split("\n")
    prefix = " " * indent
    extra_indent = " " * len(label)
    print(f"{prefix}{label}{lines[0]}")
    for line in lines[1:]:
        print(f"{prefix}{extra_indent}{line}")


# Simulate a small solution tensor: (N=3 timesteps, B=2 batches, S=2 states)
N, B, S = 3, 2, 2
y = np.arange(N * B * S).reshape(N, B, S).astype(float)

print("Original tensor y, shape (N, B, S) =", y.shape)
for t in range(N):
    print_array_aligned(f"t={t}: ", y[t])

# --- OLD: triple-nested Python loop ---
rows: list[dict[str, object]] = []
for batch_idx in range(B):
    for time_idx in range(N):
        row: dict[str, object] = {"id": batch_idx, "time": time_idx}
        for state_idx in range(S):
            row[f"state_{state_idx}"] = y[time_idx, batch_idx, state_idx]
        rows.append(row)

df_loop = pd.DataFrame(rows)
print("\n--- Loop-based DataFrame ---")
print(df_loop)

# --- NEW: vectorized ---

# Step 1: transpose (N, B, S) -> (B, N, S)
#   Groups all timesteps for each batch together
y_transposed = y.transpose(1, 0, 2)
print("\nAfter transpose (B, N, S):")
for b in range(B):
    print_array_aligned(f"batch={b}: ", y_transposed[b])

# Step 2: reshape (B, N, S) -> (B*N, S)
#   Flattens batch and time into a single row dimension
y_flat = y_transposed.reshape(-1, S)
print(f"\nAfter reshape (-1, S) = {y_flat.shape}:")
print("Y flat:")
print(y_flat)

# Step 3: build id and time columns
print("\n" + "-" * 60)
print("np.repeat() vs np.tile():")
print("-" * 60)

example = np.array([0, 1])
print(f"  Original: {example}")
print(f"  np.repeat({example}, 3) = {np.repeat(example, 3)}")
print("    ↑ Repeats EACH element 3 times")
print(f"  np.tile({example}, 3)   = {np.tile(example, 3)}")
print("    ↑ Repeats WHOLE array 3 times")

ids = np.repeat(np.arange(B), N)  # [0,0,0, 1,1,1]
times = np.tile(np.arange(N), B)  # [0,1,2, 0,1,2]
print("\nFor our DataFrame:")
print(f"  ids   (repeat batch indices): {ids}")
print(f"  times (tile time indices):    {times}")

# Step 4: assemble DataFrame
df_data: dict[str, object] = {"id": ids, "time": times}
for s in range(S):
    df_data[f"state_{s}"] = y_flat[:, s]

df_vec = pd.DataFrame(df_data)
print("\n--- Vectorized DataFrame ---")
print(df_vec)

# Verify identical
assert df_loop.equals(df_vec), "DataFrames differ!"
print("\n✓ Both DataFrames are identical.")

# --- Explanation: reshape vs stacking ---
print("\n" + "=" * 60)
print("Q: Is reshape just stacking along a dimension?")
print("=" * 60)

print("\nA: No. Reshape just reinterprets memory boundaries.")
print("   It doesn't move data (in C-contiguous arrays).\n")

# Show that reshape is a VIEW (shares memory)
print("Before transpose, y is C-contiguous (row-major):")
print(f"  y.flags.c_contiguous = {y.flags.c_contiguous}")
print(f"  Flat memory order: {y.ravel()}")

y_transposed = y.transpose(1, 0, 2)
print("\nAfter transpose, memory is reorganized:")
print(f"  y_transposed.flags.c_contiguous = {y_transposed.flags.c_contiguous}")
print(f"  Flat memory order: {y_transposed.ravel()}")

y_flat = y_transposed.reshape(-1, S)
print(f"\nAfter reshape to ({B * N}, {S}):")
print(f"  y_flat.flags.c_contiguous = {y_flat.flags.c_contiguous}")
print(f"  Flat memory order: {y_flat.ravel()}")
print("  ↑ Same as y_transposed — reshape just changed dimension labels")

print("\nIn contrast, stacking operations COPY data:")
batch0 = y_transposed[0]  # shape (N, S)
batch1 = y_transposed[1]  # shape (N, S)
y_stacked = np.vstack([batch0, batch1])  # concatenates
print(f"  np.vstack creates new array: {y_stacked.ravel()}")
print("  Result identical to reshape, but vstack copied data.")

print("\nKey insight:")
print("  transpose() reorganizes memory: (N,B,S) → (B,N,S)")
print("  reshape() reinterprets it as: batch0_times, batch1_times, ...")
print("  This 'unrolls' batches sequentially without copying.")
