
import torch
import numpy as np
from Solution import Solution
from FeatureExtractor import PendulumFeatureExtractor


def demo_pendulum_extractor():
    # N=100, B=3, S=2 (e.g., angle and angular velocity)
    N, B, S = 10, 3, 2

    # Create random data for demonstration
    initial_conditions = torch.rand((B, S))
    time_points = torch.linspace(0, 10, steps=N)
    # shape: (N, B, S)
    trajectory = torch.randn(N, B, S)

    # Create a mock Solution
    sol = Solution(
        initial_condition=initial_conditions,
        time=time_points,
        y=trajectory,
    )

    print(trajectory)
    print(trajectory[:, :, 1].shape)
    print(trajectory[:, :, 1])

    # Use PendulumFeatureExtractor
    extractor = PendulumFeatureExtractor(time_steady=2.0)
    features = extractor.extract_features(sol)

    print("Shape of original trajectory:", trajectory.shape)  # (N, B, S)
    # (B, 2) for PendulumOHE
    print("Features shape:", features.shape)


if __name__ == "__main__":
    demo_pendulum_extractor()
