This markdown document contains all the necessary parameters, equations, and methodological details from the sources to recreate the Rössler attractor basin stability experiment in Python.

---

# Case Study: Basin Stability in Rössler Networks

This document details the parameters and methodology required to investigate the **Basin Stability ($S_B$)** of the synchronous state in a network of coupled Rössler oscillators, as described in the sources.

## 1. Mathematical Model

The dynamics of the system are governed by $N$ identical Rössler oscillators coupled through their $x$-components.

### Ordinary Differential Equations (ODEs)

For each node $i = 1, \dots, N$:

- $\dot{x}_i = -y_i - z_i - K \sum_{j=1}^N L_{ij}x_j$
- $\dot{y}_i = x_i + ay_i$
- $\dot{z}_i = b + z_i(x_i - c)$

### Fixed Parameters

- **Oscillator Constants:** $a = 0.2, b = 0.2, c = 7.0$.
- **Laplacian Matrix ($L$):** Defined as $L_{ij} = \delta_{ij} \sum_k A_{ik} - A_{ij}$.
- **Coupling Constant ($K$):** Values must be selected from the stability interval $I_s$.

## 2. Network Topology

The specific network configuration provided in the edge list image is a **Watts-Strogatz (WS) network**.

- **Network Size ($N$):** 100 nodes.
- **Edges ($E$):** 400 edges.
- **Rewiring Probability ($p$):** 0.2.

### Full Edge List

For implementation in Python, use the following connections extracted from the sources:
`0-1, 0-2, 0-3, 4-61, 2-91, 1-3, 1-4, 1-5, 3-20, 2-4, 2-5, 2-6, 3-4, 3-5, 3-6, 3-7, 4-5, 4-6, 7-73, 4-8, 5-6, 5-7, 5-8, 5-9, 7-88, 8-45, 6-9, 6-10, 7-8, 7-9, 7-10, 7-11, 8-9, 8-10, 8-11, 8-12, 9-10, 9-11, 9-12, 9-13, 10-11, 10-12, 10-13, 10-14, 12-93, 11-13, 14-3, 11-15, 12-13, 12-14, 12-15, 12-16, 14-49, 13-15, 13-16, 13-17, 14-15, 14-16, 14-17, 14-18, 15-16, 15-17, 15-18, 19-65, 16-17, 16-18, 16-19, 20-94, 17-18, 17-19, 17-20, 17-21, 18-19, 20-65, 18-21, 18-22, 19-20, 19-21, 19-22, 19-23, 20-21, 22-16, 20-23, 20-24, 21-22, 21-23, 24-0, 21-25, 22-23, 22-24, 22-25, 22-26, 24-47, 23-25, 23-26, 27-75, 24-25, 26-34, 27-17, 28-90, 25-26, 25-27, 28-8, 25-29, 26-27, 26-28, 26-29, 26-30, 27-28, 27-29, 27-30, 27-31, 28-29, 30-46, 31-36, 28-32, 30-70, 29-31, 29-32, 29-33, 30-31, 30-32, 30-33, 30-34, 31-32, 31-33, 31-34, 35-52, 32-33, 32-34, 35-6, 36-48, 33-34, 33-35, 33-36, 33-37, 34-35, 34-36, 34-37, 34-38, 35-36, 35-37, 35-38, 35-39, 36-37, 36-38, 39-73, 36-40, 37-38, 37-39, 40-84, 37-41, 38-39, 38-40, 38-41, 42-93, 39-40, 41-46, 39-42, 43-13, 40-41, 40-42, 43-86, 40-44, 41-42, 41-43, 41-44, 45-67, 42-43, 42-44, 45-77, 46-95, 43-44, 43-45, 43-46, 43-47, 44-45, 44-46, 44-47, 44-48, 45-46, 47-60, 45-48, 49-75, 46-47, 46-48, 46-49, 46-50, 47-48, 47-49, 47-50, 51-96, 48-49, 48-50, 48-51, 52-34, 49-50, 49-51, 49-52, 49-53, 50-51, 50-52, 50-53, 50-54, 51-52, 51-53, 51-54, 51-55, 53-28, 54-21, 52-55, 56-83, 54-1, 53-55, 56-88, 57-88, 55-79, 54-56, 54-57, 58-78, 56-85, 55-57, 55-58, 55-59, 56-57, 56-58, 56-59, 56-60, 57-58, 57-59, 60-45, 57-61, 58-59, 58-60, 61-36, 62-75, 60-71, 59-61, 59-62, 63-12, 61-33, 60-62, 60-63, 64-7, 61-62, 61-63, 61-64, 61-65, 62-63, 62-64, 62-65, 66-77, 63-64, 63-65, 66-74, 67-47, 64-65, 64-66, 67-95, 68-19, 65-66, 65-67, 65-68, 65-69, 66-67, 66-68, 66-69, 66-70, 67-68, 67-69, 67-70, 67-71, 68-69, 68-70, 68-71, 68-72, 69-70, 69-71, 69-72, 73-14, 70-71, 70-72, 73-18, 74-16, 71-72, 71-73, 71-74, 71-75, 72-73, 72-74, 72-75, 72-76, 74-47, 73-75, 73-76, 77-56, 75-10, 74-76, 74-77, 74-78, 75-76, 75-77, 75-78, 75-79, 77-10, 76-78, 79-97, 76-80, 77-78, 77-79, 77-80, 77-81, 78-79, 78-80, 81-93, 78-82, 79-80, 79-81, 79-82, 79-83, 80-81, 80-82, 83-30, 80-84, 81-82, 81-83, 84-14, 81-85, 82-83, 82-84, 82-85, 86-52, 84-50, 83-85, 83-86, 83-87, 85-51, 86-72, 84-87, 84-88, 85-86, 85-87, 85-88, 85-89, 86-87, 86-88, 86-89, 86-90, 87-88, 87-89, 87-90, 91-12, 88-89, 88-90, 88-91, 88-92, 90-97, 91-25, 89-92, 89-93, 91-10, 92-65, 90-93, 90-94, 92-54, 91-93, 94-15, 91-95, 92-93, 92-94, 92-95, 92-96, 93-94, 93-95, 93-96, 93-97, 95-18, 94-96, 94-97, 94-98, 95-96, 95-97, 95-98, 99-59, 96-97, 96-98, 96-99, 0-34, 98-60, 97-99, 97-0, 97-1, 98-99, 98-0, 98-1, 98-2, 99-0, 99-1, 99-2, 99-3`.

## 3. Stability and Constraints

To ensure linear stability, the coupling constant $K$ must be selected based on the **Master Stability Function (MSF)**.

- **MSF Thresholds:** $\alpha_1 = 0.1232$ and $\alpha_2 = 4.663$.
- **Eigenvalues of $L$ (calculated for this network):** $\lambda_{min} = 1.236$ and $\lambda_{max} = 13.87125$.
- **Stability Interval ($I_s$):** $(\alpha_1/\lambda_{min}, \alpha_2/\lambda_{max}) \approx (0.100, 0.336)$.

## 4. Basin Stability Estimation Methodology

To calculate $S_B$, the researchers used a probabilistic volume-based approach.

### State Space Subset ($Q$)

Initial conditions are drawn uniformly at random from the volume $Q = q^N$:

- **$q$ (Range for each oscillator):** $[-15, 15] \times [-15, 15] \times [-5, 35]$.

### Sampling and Success Criteria

1.  **Trial Size ($T$):** Integrate $T = 500$ random initial conditions for each $K$.
2.  **Success ($M$):** Count how many trials converge to the **synchronous state**.
3.  **Calculation:** $S_B = M/T$.

## 5. Experimental Reference Data

The researchers measured $S_B$ at 11 equally spaced values of $K$ within $I_s$ for the provided network:

| $K$ value | Measured $S_B$ |
| :-------- | :------------- |
| 0.119     | 0.226          |
| 0.139     | 0.274          |
| 0.159     | 0.330          |
| 0.179     | 0.346          |
| 0.198     | 0.472          |
| 0.218     | 0.496          |
| 0.238     | 0.594          |
| 0.258     | 0.628          |
| 0.278     | 0.656          |
| 0.297     | 0.694          |
| 0.317     | 0.690          |

**Result:** The expected **mean basin stability ($\bar{S}_B$)** for this specific 100-node network is approximately **0.49**.
