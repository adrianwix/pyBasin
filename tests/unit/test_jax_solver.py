from typing import TypedDict

import jax.numpy as jnp
import pytest
import torch
from jax import Array

from pybasin.jax_ode_system import JaxODESystem
from pybasin.solvers import JaxSolver


class ExponentialParams(TypedDict):
    decay: float


class ExponentialDecayJaxODE(JaxODESystem[ExponentialParams]):
    def ode(self, t: Array, y: Array) -> Array:
        return self.params["decay"] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


@pytest.fixture
def simple_jax_ode() -> ExponentialDecayJaxODE:
    params: ExponentialParams = {"decay": -1.0}
    return ExponentialDecayJaxODE(params)


def test_jax_solver_integration(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[-1].item() < y[0].item()


def test_jax_solver_batched(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)

    y0 = torch.tensor([[1.0], [2.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 2, 1)
    assert torch.allclose(y[:, 1, :] / y[:, 0, :], torch.tensor([[2.0]]), atol=1e-5)  # type: ignore[misc]


def test_jax_solver_y0_shape_validation(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)

    y0_1d = torch.tensor([1.0])
    with pytest.raises(ValueError, match="y0 must be 2D with shape"):
        solver.integrate(simple_jax_ode, y0_1d)

    y0_3d = torch.tensor([[[1.0]]])
    with pytest.raises(ValueError, match="y0 must be 2D with shape"):
        solver.integrate(simple_jax_ode, y0_3d)

    y0_2d = torch.tensor([[1.0]])
    _, y_result = solver.integrate(simple_jax_ode, y0_2d)
    assert y_result.shape == (10, 1, 1)


def test_jax_solver_custom_solver(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    from diffrax import Tsit5

    custom_solver = Tsit5()
    solver = JaxSolver(
        time_span=(0, 1), n_steps=10, device="cpu", solver=custom_solver, use_cache=False
    )

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[-1].item() < y[0].item()


def test_jax_solver_with_device(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), n_steps=10, device="cpu", use_cache=False)
    new_solver = solver.with_device("cpu")

    assert new_solver is not solver
    assert new_solver.time_span == solver.time_span
    assert new_solver.n_steps == solver.n_steps
    assert new_solver.rtol == solver.rtol
    assert new_solver.atol == solver.atol

    y0 = torch.tensor([[1.0]])
    t, y = new_solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)


def test_jax_solver_default_n_steps(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), device="cpu", use_cache=False)

    assert solver.n_steps == 500

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (500,)
    assert y.shape == (500, 1, 1)


def test_jax_solver_2d_system() -> None:
    class LorenzLikeODE(JaxODESystem[dict[str, float]]):
        def ode(self, t: Array, y: Array) -> Array:
            x, v = y[..., 0], y[..., 1]
            dx = v
            dv = -x
            return jnp.stack([dx, dv], axis=-1)

        def get_str(self) -> str:
            return "harmonic_oscillator"

    ode = LorenzLikeODE({})
    solver = JaxSolver(time_span=(0, 2 * 3.14159), n_steps=100, device="cpu", use_cache=False)

    y0 = torch.tensor([[1.0, 0.0]])
    t, y = solver.integrate(ode, y0)

    assert t.shape == (100,)
    assert y.shape == (100, 1, 2)
    assert y[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[0, 0, 1].item() == pytest.approx(0.0, abs=1e-5)  # type: ignore[misc]
