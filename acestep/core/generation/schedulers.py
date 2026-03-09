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
    """Log-SNR uniform — uniform spacing in logit(t) space.

    In flow matching, SNR = (1-t)²/t², so log-SNR ∝ -logit(t).
    Uniform spacing in logit(t) concentrates steps where the
    signal-to-noise ratio changes most in *relative* terms.

    Gives an S-shaped distribution: dense around t=0.5, tapering
    at both extremes.  Preserves vocal detail budget.
    """
    # logit bounds: t=0.9986 → logit≈6.57,  t=0.0014 → logit≈-6.57
    t_max = 0.9986
    t_min = 0.0014
    logit_max = math.log(t_max / (1.0 - t_max))
    logit_min = math.log(t_min / (1.0 - t_min))

    raw = []
    for i in range(num_steps):
        frac = i / num_steps
        logit_t = logit_max + (logit_min - logit_max) * frac
        t = 1.0 / (1.0 + math.exp(-logit_t))  # sigmoid
        raw.append(t)

    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def sgm_uniform_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Karras σ-ramp — uniform in σ^(1/ρ) space (EDM/Karras convention).

    Uses the well-tested Karras noise schedule ramp with ρ=7, adapted
    for flow matching.  Provides moderate front-loading: more steps in
    the structural region than linear, but not so extreme that detail
    is starved.
    """
    t_max = 0.999
    t_min = 0.001
    sigma_max = t_max / (1.0 - t_max)   # ≈999
    sigma_min = t_min / (1.0 - t_min)   # ≈0.001
    rho = 7.0  # Karras ramp parameter

    inv_rho = 1.0 / rho
    s_max = sigma_max ** inv_rho
    s_min = sigma_min ** inv_rho

    raw = []
    for i in range(num_steps):
        frac = i / num_steps
        sigma = (s_max + frac * (s_min - s_max)) ** rho
        t = sigma / (1.0 + sigma)
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


def composite_schedule(
    num_steps: int,
    shift: float = 1.0,
    scheduler_a: str = "bong_tangent",
    scheduler_b: str = "linear",
    crossover: float = 0.5,
    split: float = 0.5,
) -> List[float]:
    """Two-stage composite — different schedulers for structure vs detail.

    Splits the total diffusion trajectory into two phases at a
    crossover timestep, using ``scheduler_a`` for the high-noise
    structural phase (t=1 → crossover) and ``scheduler_b`` for
    the low-noise detail phase (crossover → 0).

    Args:
        num_steps:   Total number of diffusion steps.
        shift:       Timestep shift factor (applied after compositing).
        scheduler_a: Scheduler name for the structural phase (t ≥ crossover).
        scheduler_b: Scheduler name for the detail phase (t < crossover).
        crossover:   Timestep value (0–1) where the phases meet. Default 0.5.
        split:       Fraction of total steps allocated to phase A. Default 0.5.
    """
    n_a = max(int(num_steps * split), 1)
    n_b = max(num_steps - n_a, 1)

    # Generate a full schedule from each scheduler, then filter to the
    # appropriate range.  We over-sample and pick the right density.
    fn_a = SCHEDULERS.get(scheduler_a, linear_schedule)
    fn_b = SCHEDULERS.get(scheduler_b, linear_schedule)

    # Phase A: structural (t=1 → crossover), n_a steps
    # Generate with shift=1 (we apply shift globally at the end)
    full_a = fn_a(n_a * 4, shift=1.0)  # oversample for better density
    phase_a = [t for t in full_a if t >= crossover]
    # If oversampled, subsample to n_a steps evenly
    if len(phase_a) > n_a:
        indices = [int(i * (len(phase_a) - 1) / (n_a - 1)) for i in range(n_a)]
        phase_a = [phase_a[idx] for idx in indices]
    elif len(phase_a) < n_a:
        # Fallback: linearly space between 1.0 and crossover
        phase_a = [1.0 - i * (1.0 - crossover) / n_a for i in range(n_a)]

    # Phase B: detail (crossover → 0), n_b steps
    full_b = fn_b(n_b * 4, shift=1.0)  # oversample
    phase_b = [t for t in full_b if t < crossover]
    if len(phase_b) > n_b:
        indices = [int(i * (len(phase_b) - 1) / (n_b - 1)) for i in range(n_b)]
        phase_b = [phase_b[idx] for idx in indices]
    elif len(phase_b) < n_b:
        # Fallback: linearly space between crossover and near-0
        phase_b = [crossover * (1.0 - (i + 1) / n_b) for i in range(n_b)]

    raw = phase_a + phase_b
    raw = sorted(raw, reverse=True)
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
    "composite": composite_schedule,
}

SCHEDULER_INFO = {
    "linear":           {"name": "Linear",           "description": "Uniform spacing (default)"},
    "ddim_uniform":     {"name": "DDIM Uniform",     "description": "Log-SNR uniform (S-shaped)"},
    "sgm_uniform":      {"name": "SGM Uniform",      "description": "Karras σ-ramp (ρ=7)"},
    "bong_tangent":     {"name": "Tangent",           "description": "Front-loaded (structural focus)"},
    "linear_quadratic": {"name": "Linear-Quadratic",  "description": "Linear start, quadratic finish"},
    "composite":        {"name": "Composite",         "description": "Two-stage: different schedulers for structure vs detail"},
}

VALID_SCHEDULERS = set(SCHEDULERS.keys())


def get_schedule(name: str, num_steps: int, shift: float = 1.0) -> List[float]:
    """Get a timestep schedule by name, supporting composite syntax.

    Simple schedulers::

        get_schedule("linear", 50, 5.0)
        get_schedule("bong_tangent", 50, 5.0)

    Composite (2-stage) scheduler — encode config in the name string::

        get_schedule("composite:bong_tangent+linear_quadratic:0.5:0.6", 50, 5.0)
        #            ^type     ^stageA     ^stageB             ^cross ^split

    Format: ``composite:<scheduler_a>+<scheduler_b>:<crossover>:<split>``

    - scheduler_a: structural phase (high noise). Default: bong_tangent
    - scheduler_b: detail phase (low noise). Default: linear
    - crossover:   timestep (0–1) separating phases. Default: 0.5
    - split:       fraction of steps for phase A. Default: 0.5

    Returns:
        List of float timestep values, descending, in (0, 1].
    """
    name = name.lower().strip()

    # ── Composite scheduler parsed from string ───────────────────
    if name.startswith("composite"):
        # Parse: "composite:a+b:crossover:split"
        parts = name.split(":")
        sched_pair = parts[1] if len(parts) > 1 else "bong_tangent+linear"
        if "+" in sched_pair:
            a_name, b_name = sched_pair.split("+", 1)
        else:
            a_name, b_name = sched_pair, "linear"
        crossover = float(parts[2]) if len(parts) > 2 else 0.5
        split_frac = float(parts[3]) if len(parts) > 3 else 0.5
        return composite_schedule(
            num_steps, shift,
            scheduler_a=a_name.strip(),
            scheduler_b=b_name.strip(),
            crossover=crossover,
            split=split_frac,
        )

    # ── Simple scheduler ─────────────────────────────────────────
    if name not in SCHEDULERS:
        valid = ", ".join(sorted(VALID_SCHEDULERS))
        raise ValueError(f"Unknown scheduler '{name}'. Valid schedulers: {valid}")
    return SCHEDULERS[name](num_steps, shift)


def get_scheduler(name: str):
    """Get a scheduler function by name (legacy API).

    For composite schedulers, returns a wrapper that parses the
    composite string.  For simple schedulers, returns the function
    directly.

    Returns:
        schedule_fn: Callable[[int, float], List[float]]

    Raises:
        ValueError if the scheduler name is not recognized.
    """
    clean = name.lower().strip()
    if clean.startswith("composite"):
        # Return a closure that delegates to get_schedule
        def _composite_wrapper(num_steps: int, shift: float = 1.0) -> List[float]:
            return get_schedule(name, num_steps, shift)
        return _composite_wrapper

    if clean not in SCHEDULERS:
        valid = ", ".join(sorted(VALID_SCHEDULERS))
        raise ValueError(f"Unknown scheduler '{clean}'. Valid schedulers: {valid}")
    return SCHEDULERS[clean]
