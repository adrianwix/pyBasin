import torch
from typing import TypedDict
from pybasin.ODESystem import ODESystem


class FrictionParams(TypedDict):
    """
    Parameters for the friction ODE system.
    Adjust them as needed to match your scenario.
    """
    v_d: float     # Driving velocity
    xi: float      # Damping ratio
    musd: float    # Ratio of static to dynamic friction coefficient
    mud: float     # Dynamic friction coefficient
    muv: float     # Linear strengthening parameter
    v0: float      # Reference velocity for exponential decay


class FrictionODE(ODESystem[FrictionParams]):
    """
    ODE System for a friction-based SDOF oscillator,
    adapted from ode_friction.m.
    """

    def __init__(self, params: FrictionParams):
        super().__init__(params)

    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vectorized right-hand side (RHS) for the friction system using PyTorch.

        y shape: (..., 2) where the last dimension is [disp, vel]
        returns: tensor with the same shape as y.
        """
        # Extract parameters
        v_d = self.params["v_d"]
        xi = self.params["xi"]
        musd = self.params["musd"]
        mud = self.params["mud"]
        muv = self.params["muv"]
        v0 = self.params["v0"]

        # Unpack states
        disp = y[..., 0]
        vel = y[..., 1]

        # Relative velocity and small threshold
        vrel = vel - v_d
        eta = 1e-4

        # Friction force
        Ffric = self._friction_law(vrel, mud, musd, muv, v0)

        # Compute conditions
        slip_condition = torch.abs(vrel) > eta
        trans_condition = torch.abs(disp + 2 * xi * vel) > mud * musd

        # Vectorized computation of derivatives
        dydt = torch.zeros_like(y)

        # First component (displacement)
        dydt[..., 0] = torch.where(
            slip_condition | trans_condition,
            vel,
            torch.full_like(vel, v_d)
        )

        # Second component (velocity)
        slip_vel = -disp - 2 * xi * vel - torch.sign(vrel) * Ffric
        trans_vel = -disp - 2 * xi * vel + mud * musd * torch.sign(Ffric)
        stick_vel = -(vel - v_d)

        dydt[..., 1] = torch.where(
            slip_condition,
            slip_vel,
            torch.where(trans_condition, trans_vel, stick_vel)
        )

        return dydt

    def _friction_law(self, vrel: torch.Tensor, mud: float, musd: float, muv: float, v0: float) -> torch.Tensor:
        """
        Exponentially decaying friction plus linear strengthening term.
        """
        return (mud
                + mud * (musd - 1.0) * torch.exp(-torch.abs(vrel) / v0)
                + muv * torch.abs(vrel) / v0)

    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.

        The string is constructed using multiple line f-string interpolation.
        """
        v_d = self.params["v_d"]
        xi = self.params["xi"]
        musd = self.params["musd"]
        mud = self.params["mud"]
        muv = self.params["muv"]
        v0 = self.params["v0"]

        description = (
            f"  ddisp/dt = vel\n"
            f"  dvel/dt  = -disp - 2*({xi})*vel - sign(vel - {v_d})*friction_law(...)\n"
            f"\n"
            f"Parameters:\n"
            f"  v_d  = {v_d}\n"
            f"  xi   = {xi}\n"
            f"  musd = {musd}\n"
            f"  mud  = {mud}\n"
            f"  muv  = {muv}\n"
            f"  v0   = {v0}\n"
        )
        return description
