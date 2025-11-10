from typing import TypedDict

import pytest
import torch

from pybasin.ode_system import ODESystem
from pybasin.solver import SklearnParallelSolver, TorchDiffEqSolver, TorchOdeSolver


class ExponentialParams(TypedDict):
    decay: float


class ExponentialDecayODE(ODESystem[ExponentialParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.params["decay"] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


@pytest.fixture
def simple_ode():
    params: ExponentialParams = {"decay": -1.0}
    return ExponentialDecayODE(params)


def test_torchdiffeq_solver_integration(simple_ode):
    solver = TorchDiffEqSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 10 time evaluation points
    assert t.shape == (10,)
    # Single trajectory: 10 steps × 1 batch × 1 state
    assert y.shape == (10, 1, 1)
    # Initial condition preserved (y₀=1.0)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)
    # Exponential decay: final value < initial value
    assert y[-1].item() < y[0].item()


def test_torchode_solver_integration(simple_ode):
    solver = TorchOdeSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 10 time points
    assert t.shape == (10,)
    # Consistent shape: 10 steps × 1 batch × 1 state
    assert y.shape == (10, 1, 1)
    # Initial condition correct
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)


def test_solver_batched_integration(simple_ode):
    solver = TorchOdeSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)

    y0 = torch.tensor([[1.0], [2.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 10 time points
    assert t.shape == (10,)
    # 10 steps × 2 batches × 1 state
    assert y.shape == (10, 2, 1)
    # Ratio between trajectories maintained (y₀=[1.0] and [2.0], so ratio=2.0 throughout)
    assert torch.allclose(y[:, 1, :] / y[:, 0, :], torch.tensor([[2.0]]), atol=1e-5)


def test_solver_y0_shape_validation(simple_ode):
    solver = TorchOdeSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)

    # 1D tensor should be rejected
    y0_1d = torch.tensor([1.0])
    with pytest.raises(ValueError, match="y0 must be 2D with shape"):
        solver.integrate(simple_ode, y0_1d)

    # 3D tensor should be rejected
    y0_3d = torch.tensor([[[1.0]]])
    with pytest.raises(ValueError, match="y0 must be 2D with shape"):
        solver.integrate(simple_ode, y0_3d)

    # 2D tensor should work
    y0_2d = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0_2d)
    assert y.shape == (10, 1, 1)


def test_sklearn_solver_integration(simple_ode):
    solver = SklearnParallelSolver(time_span=(0, 1), fs=10, device="cpu", n_jobs=1, use_cache=False)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 11 time points (fs=10 means 10 samples per second, so 0 to 1 = 11 points)
    assert t.shape == (11,)
    # Single trajectory: 11 steps × 1 batch × 1 state
    assert y.shape == (11, 1, 1)
    # Initial condition preserved (y₀=1.0)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)
    # Exponential decay: final value < initial value
    assert y[-1].item() < y[0].item()


def test_sklearn_solver_batched(simple_ode):
    solver = SklearnParallelSolver(time_span=(0, 1), fs=10, device="cpu", n_jobs=2, use_cache=False)

    y0 = torch.tensor([[1.0], [2.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 11 time points
    assert t.shape == (11,)
    # 11 steps × 2 batches × 1 state
    assert y.shape == (11, 2, 1)
    # Ratio between trajectories maintained (y₀=[1.0] and [2.0], so ratio=2.0 throughout)
    assert torch.allclose(y[:, 1, :] / y[:, 0, :], torch.tensor([[2.0]]), atol=1e-5)
