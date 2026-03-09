"""
Scheduler registry for ACE-Step flow matching diffusion.

Schedulers define HOW timesteps are spaced between t=1 (noise) and t=0 (clean).
This is independent of the solver (which defines how each step is computed)
and the guidance mode (which defines how CFG is applied).

Scheduler interface:
    schedule_fn(num_steps: int, shift: float = 1.0) -> List[float]

    - num_steps:  Number of diffusion steps (N).
    - shift:      Timestep shift factor.  Applied as final warp:
                  t' = shift * t / (1 + (shift - 1) * t)

    Returns: List of N float values, descending, in (0, 1].
             Does NOT include a trailing 0.

To add a new scheduler:
    1. Define a function following the interface above
    2. Register it in the SCHEDULERS dict
    3. Add metadata to SCHEDULER_INFO
"""

import math
from typing import List


# ---------------------------------------------------------------------------
# Shift warp — shared by all schedulers
# ---------------------------------------------------------------------------

def _apply_shift(timesteps: List[float], shift: float) -> List[float]:
    """Apply the standard shift warp: t' = shift*t / (1 + (shift-1)*t).

    When shift == 1.0 this is an identity transform.
    """
    if shift == 1.0:
        return timesteps
    return [shift * t / (1.0 + (shift - 1.0) * t) for t in timesteps]


# ---------------------------------------------------------------------------
# Schedule functions
# ---------------------------------------------------------------------------

def linear_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Linear (uniform) spacing — the original ACE-Step default.

    Produces evenly-spaced timesteps from 1.0 toward 0.0.
    Equivalent to the existing ``torch.linspace(1, 0, N+1)[:-1]`` behaviour.
    """
    raw = [1.0 - i / num_steps for i in range(num_steps)]
    return _apply_shift(raw, shift)


def ddim_uniform_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """DDIM uniform — uniform spacing in σ (sigma) space.

    Maps to sigma via σ = sqrt((1-α_bar)/α_bar) where α_bar follows a
    cosine schedule, then distributes steps uniformly in σ-space and
    maps back to t.  This concentrates steps where the noise level
    changes most rapidly.
    """
    # Build a fine-grained cosine alpha_bar schedule
    N_fine = 1000
    alphas_bar = [
        math.cos(((i / N_fine) + 0.008) / 1.008 * math.pi / 2) ** 2
        for i in range(N_fine + 1)
    ]
    sigmas = [math.sqrt((1.0 - ab) / max(ab, 1e-8)) for ab in alphas_bar]
    sigma_max = sigmas[0]
    sigma_min = sigmas[-1]

    # Uniform steps in sigma space
    target_sigmas = [
        sigma_max + (sigma_min - sigma_max) * i / (num_steps)
        for i in range(num_steps)
    ]

    # Map each target sigma back to t ∈ (0, 1]
    # t = σ² / (1 + σ²), derived from σ = sqrt((1-t)/t) → t = 1/(1+σ²)
    # But we want t=1 → noise, t=0 → clean, so t = σ²/(1+σ²)
    raw = [s * s / (1.0 + s * s) for s in target_sigmas]

    # Ensure descending and clamp
    raw = sorted(raw, reverse=True)
    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def sgm_uniform_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """SGM uniform — uniform in σ² space (Stability AI / EDM convention).

    Distributes timesteps uniformly in σ² (variance) rather than σ,
    giving a subtly different density curve from DDIM uniform.
    """
    # In flow matching: t corresponds to the noise fraction, σ² ~ t/(1-t)
    # Uniform in σ² means uniform in t/(1-t), so we solve for t.
    # Let u = t/(1-t) → t = u/(1+u)
    # u ranges from large (t≈1) to small (t≈0)
    u_max = 1.0 / 1e-4 - 1.0  # t ≈ 0.9999
    u_min = 1e-4 / (1.0 - 1e-4)  # t ≈ 0.0001

    raw = []
    for i in range(num_steps):
        frac = i / num_steps
        u = u_max + (u_min - u_max) * frac
        t = u / (1.0 + u)
        raw.append(t)

    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def bong_tangent_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Tangent-based spacing — concentrates steps at high noise levels.

    Uses a tangent curve to front-load steps where the model makes
    major structural decisions, with fewer steps in the fine-detail
    region near t=0.
    """
    raw = []
    for i in range(num_steps):
        # Map i to angle in (0, π/2), tangent maps (0,π/2) → (0, ∞)
        # We then normalise to (0, 1]
        frac = (i + 0.5) / num_steps  # avoid exact 0 and 1
        angle = frac * math.pi / 2.0
        tan_val = math.tan(angle)
        # Normalise: at frac=1, tan(π/2) → ∞, so use a bounded version
        # Use atan-based mapping: t = 1 - (2/π) * atan(tan_val * scale)
        # scale controls how aggressively front-loaded the schedule is
        scale = 1.5
        t = 1.0 - (2.0 / math.pi) * math.atan(tan_val * scale)
        raw.append(t)

    # Sort descending and clamp
    raw = sorted(raw, reverse=True)
    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def linear_quadratic_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Linear start, quadratic finish — more steps for fine detail.

    The first half of steps are linearly spaced (even coverage of
    high-noise structural region), and the second half use quadratic
    spacing (denser toward t=0 for fine detail refinement).

    The crossover fraction (0.5) balances structure vs detail.
    """
    crossover = 0.5
    n_linear = max(int(num_steps * crossover), 1)
    n_quad = num_steps - n_linear

    # Linear region: from 1.0 down to crossover point
    t_cross = 1.0 - crossover  # e.g., 0.5
    linear_part = [1.0 - i * crossover / n_linear for i in range(n_linear)]

    # Quadratic region: from crossover point down toward 0
    quad_part = []
    for i in range(n_quad):
        frac = (i + 1) / n_quad  # 0 → 1, maps crossover → 0
        # Quadratic: denser at end (near 0)
        t = t_cross * (1.0 - frac ** 2)
        quad_part.append(t)

    raw = linear_part + quad_part
    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCHEDULERS = {
    "linear": linear_schedule,
    "ddim_uniform": ddim_uniform_schedule,
    "sgm_uniform": sgm_uniform_schedule,
    "bong_tangent": bong_tangent_schedule,
    "linear_quadratic": linear_quadratic_schedule,
}

SCHEDULER_INFO = {
    "linear":           {"name": "Linear",           "description": "Uniform spacing (default)"},
    "ddim_uniform":     {"name": "DDIM Uniform",     "description": "Uniform in σ-space"},
    "sgm_uniform":      {"name": "SGM Uniform",      "description": "Uniform in σ²-space (EDM)"},
    "bong_tangent":     {"name": "Tangent",           "description": "Front-loaded (structural focus)"},
    "linear_quadratic": {"name": "Linear-Quadratic",  "description": "Linear start, quadratic finish"},
}

VALID_SCHEDULERS = set(SCHEDULERS.keys())


def get_scheduler(name: str):
    """Get a scheduler function by name.

    Returns:
        schedule_fn: Callable[[int, float], List[float]]

    Raises:
        ValueError if the scheduler name is not recognized.
    """
    name = name.lower().strip()
    if name not in SCHEDULERS:
        valid = ", ".join(sorted(VALID_SCHEDULERS))
        raise ValueError(f"Unknown scheduler '{name}'. Valid schedulers: {valid}")
    return SCHEDULERS[name]
