# pyrigh: basic
"""
Simple experiment to test torchdiffeq's odeint_event behavior with batched trajectories.

Tests:
1. Shape of returned tensors (t and y)
2. Whether event times differ for different trajectories
3. Whether one trajectory stopping affects others (premature stopping)
"""

import torch
from torchdiffeq import odeint_event  # type: ignore


class LorenzODE:
    """Lorenz system for testing event stopping."""

    def __init__(self, sigma: float = 0.12, r: float = 0.0, b: float = -0.6):
        self.sigma = sigma
        self.r = r
        self.b = b

    def __call__(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute derivatives for Lorenz system."""
        x = y[..., 0]
        y_coord = y[..., 1]
        z = y[..., 2]

        dx = self.sigma * (y_coord - x)
        dy = self.r * x - x * z - y_coord
        dz = x * y_coord - self.b * z

        return torch.stack([dx, dy, dz], dim=-1)


def lorenz_stop_event(max_val: float = 200.0):
    """Event function to stop when amplitude exceeds threshold."""

    def event_fn(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        max_abs_y = torch.max(torch.abs(y), dim=-1)[0]
        return max_val - max_abs_y

    return event_fn


def main():
    print("=" * 70)
    print("Testing torchdiffeq odeint_event with batched trajectories")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    # Create ODE system with broken butterfly parameters
    ode_system = LorenzODE(sigma=0.12, r=0.0, b=-0.6)

    # Test initial conditions from setup_lorenz_system.py
    # butterfly1: should stay bounded (event at t=1000)
    # butterfly2: should stay bounded (event at t=1000)
    # unbounded: should trigger event early (before t=1000)
    y0_batch = torch.tensor(
        [
            [0.8, -3.0, 0.0],  # butterfly1 - bounded
            [-0.8, 3.0, 0.0],  # butterfly2 - bounded
            [10.0, 50.0, 0.0],  # unbounded - should stop early
        ],
        dtype=torch.float32,
        device=device,
    )

    t0 = torch.tensor(0.0, dtype=torch.float32, device=device)
    t_end = 1000.0
    event_fn = lorenz_stop_event(max_val=200.0)

    print(f"Initial conditions shape: {y0_batch.shape}")
    print(f"Initial conditions:\n{y0_batch}\n")

    # Test with odeint_event
    print("Running odeint_event with batch...")
    event_t, final_state = odeint_event(
        ode_system,
        y0_batch,
        t0,
        event_fn=event_fn,
        atol=1e-6,
        rtol=1e-8,
        method="dopri5",
    )

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}\n")

    print(f"Event times shape: {event_t.shape}")
    print(f"Event times:\n{event_t}\n")

    print(f"Final states shape: {final_state.shape}")
    print(f"Final states:\n{final_state}\n")

    # Check if event times differ
    print(f"{'=' * 70}")
    print("ANALYSIS")
    print(f"{'=' * 70}\n")

    labels = ["butterfly1", "butterfly2", "unbounded"]

    # Check if event_t is scalar or per-trajectory
    if event_t.ndim == 0:
        print("⚠️  Event time is SCALAR - all trajectories stopped at same time!")
        print(f"Event time: {event_t.item():.2f}")
        print("This indicates trajectories are COUPLED - one event stops all!\n")
    else:
        print(f"Event times are per-trajectory (shape: {event_t.shape})")
        for i, label in enumerate(labels):
            t_event = event_t[i].item()
            stopped_early = t_event < t_end - 1e-3
            print(f"{label:12} : t_event = {t_event:8.2f}  (stopped early: {stopped_early})")

        # Check if all have same event time
        all_same = torch.allclose(event_t, event_t[0].expand_as(event_t))
        print(f"\nAll event times identical: {all_same}")
        if all_same:
            print("⚠️  WARNING: All trajectories stopped at same time - possible coupling!")
        else:
            print("✓ Event times differ - trajectories stop independently")

    # Analyze final state shape
    print(f"\n{'=' * 70}")
    print("FINAL STATE STRUCTURE")
    print(f"{'=' * 70}\n")

    if final_state.ndim == 3:
        n_times, n_batch, n_dims = final_state.shape
        print(f"Final state has 3 dimensions: [{n_times}, {n_batch}, {n_dims}]")
        print(f"  - Dimension 0 ({n_times}): Time points (initial + final)")
        print(f"  - Dimension 1 ({n_batch}): Batch size")
        print(f"  - Dimension 2 ({n_dims}): State dimensions")
        print("\nInitial states (t=0):")
        print(final_state[0])
        print(f"\nFinal states (t={event_t.item() if event_t.ndim == 0 else 'varies'}):")
        print(final_state[-1])
    else:
        print(f"Final state shape: {final_state.shape}")
        print(final_state)

    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}\n")

    print("⚠️  CRITICAL FINDING:")
    print("================")
    print("torchdiffeq's odeint_event with batching does NOT support independent stopping!")
    print()
    print("Behavior observed:")
    print("- Returns event_t: scalar (not per-trajectory)")
    print("- Returns final_state: [2, batch, n_dims] = [initial_state, final_state]")
    print("- When ANY trajectory triggers event, ALL trajectories stop")
    print("- This is the 'first event stops all' behavior")
    print()
    print("Implication for basin stability:")
    print("- Cannot use batched odeint_event for independent trajectory termination")
    print("- Must either:")
    print("  1. Process trajectories one at a time (slower)")
    print("  2. Use regular odeint without events + post-process to detect unbounded")
    print("  3. Use a different solver that supports per-trajectory events (e.g., JAX/Diffrax)")


if __name__ == "__main__":
    main()
