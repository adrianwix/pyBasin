# pyright: basic
"""
Experiment: Cache serialization format comparison.

Compares three serialization methods for caching torch tensors:
1. pickle (stdlib)
2. torch.save / torch.load (PyTorch native)
3. safetensors (Hugging Face, zero-copy memory-mapped loads)

Measures save time, load time, and file size across multiple tensor sizes.
"""

import os
import pickle
import shutil
import tempfile
import time
from collections.abc import Callable

import torch
from safetensors.torch import load_file, save_file  # pyright: ignore[reportUnknownVariableType]

type BenchFn = Callable[[dict[str, torch.Tensor], str], tuple[float, float, int]]

SIZES: dict[str, tuple[int, ...]] = {
    "Small (1K)": (1000,),
    "Medium (100K)": (100_000,),
    "Large (10M)": (10_000_000,),
    "XLarge (50M)": (50_000_000,),
}

N_WARMUP = 2
N_REPEATS = 10


def _make_tensors(shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
    t: torch.Tensor = torch.linspace(0, 10, shape[0])
    y: torch.Tensor = torch.randn(shape[0], 2)
    return {"t": t, "y": y}


def bench_pickle(tensors: dict[str, torch.Tensor], tmp: str) -> tuple[float, float, int]:
    path: str = os.path.join(tmp, "data.pkl")

    for _ in range(N_WARMUP):
        with open(path, "wb") as f:
            pickle.dump(tensors, f)
        with open(path, "rb") as f:
            pickle.load(f)  # noqa: S301

    save_times: list[float] = []
    for _ in range(N_REPEATS):
        t0: float = time.perf_counter()
        with open(path, "wb") as f:
            pickle.dump(tensors, f)
        save_times.append(time.perf_counter() - t0)

    load_times: list[float] = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            pickle.load(f)  # noqa: S301
        load_times.append(time.perf_counter() - t0)

    size: int = os.path.getsize(path)
    return _median(save_times), _median(load_times), size


def bench_torch(tensors: dict[str, torch.Tensor], tmp: str) -> tuple[float, float, int]:
    path: str = os.path.join(tmp, "data.pt")

    for _ in range(N_WARMUP):
        torch.save(tensors, path)
        torch.load(path, weights_only=True)  # pyright: ignore[reportUnknownMemberType]

    save_times: list[float] = []
    for _ in range(N_REPEATS):
        t0: float = time.perf_counter()
        torch.save(tensors, path)
        save_times.append(time.perf_counter() - t0)

    load_times: list[float] = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        torch.load(path, weights_only=True)  # pyright: ignore[reportUnknownMemberType]
        load_times.append(time.perf_counter() - t0)

    size: int = os.path.getsize(path)
    return _median(save_times), _median(load_times), size


def bench_safetensors(tensors: dict[str, torch.Tensor], tmp: str) -> tuple[float, float, int]:
    path: str = os.path.join(tmp, "data.safetensors")
    contiguous: dict[str, torch.Tensor] = {k: v.contiguous() for k, v in tensors.items()}

    for _ in range(N_WARMUP):
        save_file(contiguous, path)
        load_file(path)

    save_times: list[float] = []
    for _ in range(N_REPEATS):
        t0: float = time.perf_counter()
        save_file(contiguous, path)
        save_times.append(time.perf_counter() - t0)

    load_times: list[float] = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        load_file(path)
        load_times.append(time.perf_counter() - t0)

    size: int = os.path.getsize(path)
    return _median(save_times), _median(load_times), size


def _median(values: list[float]) -> float:
    s: list[float] = sorted(values)
    n: int = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _fmt_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024**2:
        return f"{nbytes / 1024:.1f} KB"
    if nbytes < 1024**3:
        return f"{nbytes / 1024**2:.1f} MB"
    return f"{nbytes / 1024**3:.2f} GB"


def run_comparison() -> None:
    """Run the full serialization benchmark and print results."""
    bench_fns: dict[str, BenchFn] = {
        "pickle": bench_pickle,
        "torch.save": bench_torch,
        "safetensors": bench_safetensors,
    }

    for size_label, shape in SIZES.items():
        tensors: dict[str, torch.Tensor] = _make_tensors(shape)
        total_elements: int = sum(t.numel() for t in tensors.values())
        total_bytes: int = sum(t.numel() * t.element_size() for t in tensors.values())

        print(f"\n{'=' * 72}")
        print(f"  {size_label}  —  {total_elements:,} elements, {_fmt_size(total_bytes)} raw")
        print(f"{'=' * 72}")
        print(
            f"  {'Method':<16} {'Save':>12} {'Load':>12} "
            f"{'Size':>10} {'Save vs pkl':>14} {'Load vs pkl':>14}"
        )
        print(f"  {'-' * 70}")

        baseline_save: float = 0.0
        baseline_load: float = 0.0

        for method_name, fn in bench_fns.items():
            tmp: str = tempfile.mkdtemp()
            try:
                save_t, load_t, fsize = fn(tensors, tmp)
            finally:
                shutil.rmtree(tmp)

            if method_name == "pickle":
                baseline_save = save_t
                baseline_load = load_t
                save_ratio: str = "baseline"
                load_ratio: str = "baseline"
            else:
                save_ratio = (
                    f"{baseline_save / save_t:.1f}x faster"
                    if save_t < baseline_save
                    else f"{save_t / baseline_save:.1f}x slower"
                    if save_t > baseline_save
                    else "same"
                )
                load_ratio = (
                    f"{baseline_load / load_t:.1f}x faster"
                    if load_t < baseline_load
                    else f"{load_t / baseline_load:.1f}x slower"
                    if load_t > baseline_load
                    else "same"
                )

            print(
                f"  {method_name:<16} {_fmt_time(save_t):>12} {_fmt_time(load_t):>12} "
                f"{_fmt_size(fsize):>10} {save_ratio:>14} {load_ratio:>14}"
            )


if __name__ == "__main__":
    run_comparison()
