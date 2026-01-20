# Rössler Network

!!! note "Documentation in Progress"
This page is under construction.

## System Description

Network of N coupled Rössler oscillators:

$$\dot{x}_i = -y_i - z_i + \sigma \sum_j A_{ij}(x_j - x_i)$$

## Attractors

- **Synchronized**: All oscillators in phase
- **Desynchronized**: Oscillators out of phase

## Key Feature

Custom `SynchronizationFeatureExtractor` and `SynchronizationClassifier`.
