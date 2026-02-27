# pyright: basic
"""Plot the friction coefficient mu(v_rel) and the role of the reference velocity v0.

Shows:
  1. The friction curve for the default parameters used in the case study.
  2. How v0 controls the sharpness of the static-to-dynamic transition.
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Friction coefficient
# ---------------------------------------------------------------------------


def mu(v_rel: np.ndarray, mu_st: float, mu_d: float, mu_v: float, v0: float) -> np.ndarray:
    """Friction coefficient as a function of relative velocity.

    :param v_rel: Relative velocity array.
    :param mu_st: Static friction coefficient mu(0).
    :param mu_d: Dynamic friction coefficient mu(v_rel -> inf).
    :param mu_v: Viscous friction coefficient.
    :param v0: Reference velocity (transition width).
    :return: Friction coefficient values.
    """
    return mu_d + (mu_st - mu_d) * np.exp(-np.abs(v_rel) / v0) + mu_v * np.abs(v_rel) / v0


# ---------------------------------------------------------------------------
# Case-study default parameters
# ---------------------------------------------------------------------------
MU_D = 0.5
MU_SD = 2.0  # mu_st / mu_d
MU_ST = MU_SD * MU_D  # = 1.0
MU_V = 0.0
V0 = 0.5
V_D = 1.5

v_rel = np.linspace(0, 4, 500)

# ---------------------------------------------------------------------------
# Figure 1 – Default parameters with annotations
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

mu_vals = mu(v_rel, MU_ST, MU_D, MU_V, V0)
ax.plot(v_rel, mu_vals, color="C0", lw=2, label=rf"$v_0 = {V0}$  (case study)")

# Horizontal reference lines
ax.axhline(MU_ST, color="gray", ls="--", lw=1)
ax.axhline(MU_D, color="gray", ls=":", lw=1)

# Annotations
ax.annotate(
    r"$\mu_\mathrm{st} = \mu(0)$",
    xy=(0, MU_ST),
    xytext=(1.5, MU_ST + 0.06),
    arrowprops={"arrowstyle": "-", "color": "gray"},
    fontsize=9,
    color="gray",
)
ax.annotate(
    r"$\mu_d = \mu(v_\mathrm{rel} \to \infty)$",
    xy=(3.5, MU_D),
    xytext=(1.8, MU_D - 0.08),
    arrowprops={"arrowstyle": "-", "color": "gray"},
    fontsize=9,
    color="gray",
)

# Mark v0 on x-axis
ax.axvline(V0, color="C1", ls="--", lw=1, label=rf"$v_0 = {V0}$")
ax.axvline(3 * V0, color="C2", ls="--", lw=1, label=rf"$3\,v_0 = {3 * V0}$  (95% transition)")

mu_at_v0 = mu(np.array([V0]), MU_ST, MU_D, MU_V, V0)[0]
mu_at_3v0 = mu(np.array([3 * V0]), MU_ST, MU_D, MU_V, V0)[0]
ax.plot(V0, mu_at_v0, "o", color="C1", ms=6, zorder=5)
ax.plot(3 * V0, mu_at_3v0, "o", color="C2", ms=6, zorder=5)

ax.set_xlabel(r"$|v_\mathrm{rel}|$")
ax.set_ylabel(r"$\mu(v_\mathrm{rel})$")
ax.set_title("Friction coefficient (default case-study parameters)")
ax.legend(fontsize=9)
ax.set_xlim(0, 4)
ax.set_ylim(0, 1.3)
fig.tight_layout()
fig.savefig("friction_coefficient_default.pdf")
print("Saved friction_coefficient_default.pdf")

# ---------------------------------------------------------------------------
# Figure 2 – Effect of v0 on the transition width
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

v0_values = [0.1, 0.5, 1.5, 4.0]
for v0_val in v0_values:
    ax.plot(v_rel, mu(v_rel, MU_ST, MU_D, MU_V, v0_val), lw=2, label=rf"$v_0 = {v0_val}$")

ax.axhline(MU_ST, color="gray", ls="--", lw=1, label=r"$\mu_\mathrm{st}$")
ax.axhline(MU_D, color="gray", ls=":", lw=1, label=r"$\mu_d$")

ax.set_xlabel(r"$|v_\mathrm{rel}|$")
ax.set_ylabel(r"$\mu(v_\mathrm{rel})$")
ax.set_title(r"Effect of reference velocity $v_0$ on the transition width")
ax.legend(fontsize=9)
ax.set_xlim(0, 4)
ax.set_ylim(0, 1.3)
fig.tight_layout()
fig.savefig("friction_coefficient_v0_comparison.pdf")
print("Saved friction_coefficient_v0_comparison.pdf")

plt.show()
