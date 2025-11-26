
# ✅ **DDSP QUADRATURE OSC CORE — FULL SPEC**

## MODULE NAME:

**ddsp_quadrature_osc_core**

---

## DESCRIPTION:

A **pure functional**, **differentiable**, **jit-stable** JAX quadrature oscillator generating **sin + cos** simultaneously from a single normalized phase input.

It is phase-driven (no internal phase accumulator) and ideal for:

* analytic signal construction
* Hilbert-like quadrature synthesis
* vector modulation (FM/PM/complex-domain processing)
* IQ modulation
* Gabor-style units in neural DSP pipelines

Includes:

* optional amplitude smoothing
* optional soft bandlimit shaping
* separate phase offsets for sine and cosine
* stateless except for smoothing

---

## INPUTS:

* **phase** : normalized incoming phase `[0,1)` (or any real; wrapped internally)
* **params.amp_target** : target amplitude
* **params.amp_smooth_coef** : smoothing α
* **params.sin_phase_offset** : normalized phase offset for sine
* **params.cos_phase_offset** : normalized phase offset for cosine
* **params.bandlimit_flag** : 0.0 = no shaping, 1.0 = apply shaping
* **params.shape_amount** : amount of quadratic soft shaping

---

## OUTPUTS:

* **y_sin** : sine output sample
* **y_cos** : cosine output sample
* **new_state** : updated smoothing state

---

## STATE VARIABLES:

```python
(
    amp_smooth,   # amplitude smoothing state
)
```

---

## EQUATIONS / MATH:

### Phase wrapping & offsets

```
p = phase - floor(phase)

p_sin = p + sin_phase_offset
p_sin = p_sin - floor(p_sin)

p_cos = p + cos_phase_offset
p_cos = p_cos - floor(p_cos)
```

### Raw quadrature components

```
s0 = sin(2π * p_sin)
c0 = cos(2π * p_cos)
```

### Optional shaping

```
s_shaped = s0 * (1 - shape_amount * p_sin^2)
c_shaped = c0 * (1 - shape_amount * p_cos^2)

y_sin = s0*(1-bandlimit_flag) + s_shaped*bandlimit_flag
y_cos = c0*(1-bandlimit_flag) + c_shaped*bandlimit_flag
```

### Amplitude smoothing

```
amp_smooth[n+1] = amp_smooth[n] + α*(amp_target - amp_smooth[n])
```

### Final outputs

```
y_sin_out = amp_smooth_next * y_sin
y_cos_out = amp_smooth_next * y_cos
```

---

## NOTES:

* No Python branching inside jit
* Only JAX ops (`jnp.where`, `lax.cond`)
* No dynamic allocation inside jit
* All arrays/scalars must be JAX types
* Fully differentiable

---

# ✅ **FULL PYTHON MODULE — `ddsp_quadrature_osc_core.py`**

```python
"""
ddsp_quadrature_osc_core.py

GammaJAX DDSP — Quadrature Oscillator Core
------------------------------------------

Produces coherent sine and cosine signals from a normalized phase input.

Features:
- Consumes external phase (e.g., from phasor_osc_core)
- Outputs (sin, cos) simultaneously
- Optional amplitude smoothing
- Optional quadratic shaping for bandlimit softening
- Pure JAX, no side effects, no classes, fully differentiable
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple


# =============================================================================
# INIT
# =============================================================================

def ddsp_quadrature_osc_core_init(
    initial_amp: float = 1.0,
    amp_smooth_coef: float = 0.0,
    sin_phase_offset: float = 0.0,
    cos_phase_offset: float = 0.0,
    bandlimit_flag: float = 0.0,
    shape_amount: float = 0.0,
    *,
    dtype=jnp.float32,
):
    """
    Initialize quadrature oscillator state + params.

    Returns:
        state  : (amp_smooth,)
        params : (amp_target, amp_smooth_coef,
                  sin_phase_offset, cos_phase_offset,
                  bandlimit_flag, shape_amount)
    """
    amp_smooth = jnp.asarray(initial_amp, dtype=dtype)

    p_amp = jnp.asarray(initial_amp, dtype=dtype)
    p_smooth = jnp.asarray(amp_smooth_coef, dtype=dtype)
    p_sin_off = jnp.asarray(sin_phase_offset, dtype=dtype)
    p_cos_off = jnp.asarray(cos_phase_offset, dtype=dtype)
    p_bl = jnp.asarray(bandlimit_flag, dtype=dtype)
    p_shape = jnp.asarray(shape_amount, dtype=dtype)

    state = (amp_smooth,)
    params = (p_amp, p_smooth, p_sin_off, p_cos_off, p_bl, p_shape)
    return state, params


# =============================================================================
# UPDATE STATE (no separate update)
# =============================================================================

def ddsp_quadrature_osc_core_update_state(state, params):
    """Return state unchanged (all updates happen in tick)."""
    del params
    return state


# =============================================================================
# TICK
# =============================================================================

@jax.jit
def ddsp_quadrature_osc_core_tick(
    phase: jnp.ndarray,
    state: Tuple[jnp.ndarray],
    params: Tuple[jnp.ndarray, ...],
):
    """
    Single-sample quadrature oscillator tick.

    Args:
        phase  : normalized phase input
        state  : (amp_smooth,)
        params : (amp_target, amp_smooth_coef,
                  sin_phase_offset, cos_phase_offset,
                  bandlimit_flag, shape_amount)

    Returns:
        (y_sin, y_cos), new_state
    """
    (amp_smooth,) = state
    amp_target, amp_smooth_coef, sin_phase_offset, cos_phase_offset, bandlimit_flag, shape_amount = params

    # Promote/dtype-correct
    phase = jnp.asarray(phase, dtype=amp_smooth.dtype)

    amp_smooth_coef = jnp.clip(jnp.asarray(amp_smooth_coef, dtype=phase.dtype), 0.0, 1.0)
    amp_target = jnp.asarray(amp_target, dtype=phase.dtype)
    sin_phase_offset = jnp.asarray(sin_phase_offset, dtype=phase.dtype)
    cos_phase_offset = jnp.asarray(cos_phase_offset, dtype=phase.dtype)
    bandlimit_flag = jnp.clip(jnp.asarray(bandlimit_flag, dtype=phase.dtype), 0.0, 1.0)
    shape_amount = jnp.asarray(shape_amount, dtype=phase.dtype)

    # 1. Amplitude smoothing
    amp_smooth_next = amp_smooth + amp_smooth_coef * (amp_target - amp_smooth)

    # Wrap phase
    p = phase - jnp.floor(phase)

    # Sine branch phase
    p_sin = p + sin_phase_offset
    p_sin = p_sin - jnp.floor(p_sin)

    # Cosine branch phase
    p_cos = p + cos_phase_offset
    p_cos = p_cos - jnp.floor(p_cos)

    # Core trig
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=p.dtype)
    s0 = jnp.sin(two_pi * p_sin)
    c0 = jnp.cos(two_pi * p_cos)

    # Soft shaping
    shape_sin = 1.0 - shape_amount * (p_sin * p_sin)
    shape_cos = 1.0 - shape_amount * (p_cos * p_cos)

    s_shaped = s0 * shape_sin
    c_shaped = c0 * shape_cos

    y_sin = s0 * (1.0 - bandlimit_flag) + s_shaped * bandlimit_flag
    y_cos = c0 * (1.0 - bandlimit_flag) + c_shaped * bandlimit_flag

    # Amplitude apply
    y_sin_out = amp_smooth_next * y_sin
    y_cos_out = amp_smooth_next * y_cos

    new_state = (amp_smooth_next,)
    return (y_sin_out, y_cos_out), new_state


# =============================================================================
# PROCESS
# =============================================================================

@jax.jit
def ddsp_quadrature_osc_core_process(
    phase_buf: jnp.ndarray,
    state: Tuple[jnp.ndarray],
    params: Tuple[jnp.ndarray, ...],
):
    """
    Process a time buffer of phases → (sin, cos) quadrature.
    """
    phase_buf = jnp.asarray(phase_buf)
    T = phase_buf.shape[0]

    amp_target, amp_smooth_coef, sin_off, cos_off, bl_flag, shape_amt = params

    # Broadcast parameters to (T,)
    amp_target = jnp.broadcast_to(jnp.asarray(amp_target), (T,))
    amp_smooth_coef = jnp.broadcast_to(jnp.asarray(amp_smooth_coef), (T,))
    sin_off = jnp.broadcast_to(jnp.asarray(sin_off), (T,))
    cos_off = jnp.broadcast_to(jnp.asarray(cos_off), (T,))
    bl_flag = jnp.broadcast_to(jnp.asarray(bl_flag), (T,))
    shape_amt = jnp.broadcast_to(jnp.asarray(shape_amt), (T,))

    xs = (phase_buf, amp_target, amp_smooth_coef, sin_off, cos_off, bl_flag, shape_amt)

    def body(carry, xs_t):
        st = carry
        (phase_t, a_t, s_t, sin_o, cos_o, bl, sh) = xs_t
        (ys, yc), st_next = ddsp_quadrature_osc_core_tick(
            phase_t, st, (a_t, s_t, sin_o, cos_o, bl, sh)
        )
        return st_next, (ys, yc)

    final_state, outputs = lax.scan(body, state, xs)
    y_sin_buf, y_cos_buf = outputs
    return (y_sin_buf, y_cos_buf), final_state


# =============================================================================
# SMOKE TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    print("=== ddsp_quadrature_osc_core: smoke test ===")

    sr = 48000
    dur = 0.005
    T = int(sr * dur)
    freq = 440.0

    t = jnp.linspace(0.0, dur, T, endpoint=False)
    phase_buf = jnp.mod(freq * t, 1.0)

    state, params = ddsp_quadrature_osc_core_init(
        initial_amp=1.0,
        amp_smooth_coef=0.01,
        sin_phase_offset=0.0,
        cos_phase_offset=0.25,
        bandlimit_flag=0.2,
        shape_amount=0.3,
    )

    (y_sin, y_cos), _ = ddsp_quadrature_osc_core_process(phase_buf, state, params)

    y_sin_np = onp.asarray(y_sin)
    y_cos_np = onp.asarray(y_cos)

    plt.plot(y_sin_np[:300], label="sin")
    plt.plot(y_cos_np[:300], label="cos")
    plt.legend()
    plt.title("Quadrature Oscillator — First 300 Samples")
    plt.show()

    # Optional Lissajous curve
    plt.plot(y_cos_np[:1000], y_sin_np[:1000])
    plt.title("Quadrature Lissajous Plot")
    plt.xlabel("cos")
    plt.ylabel("sin")
    plt.axis("equal")
    plt.show()

    if HAVE_SD:
        print("Playing quadrature sin only...")
        sd.play(y_sin_np * 0.2, samplerate=sr, blocking=True)
```

---

# 🎯 **Next Modules?**

You may now request:

➡ **ddsp_triangle_osc_core.py**
➡ **ddsp_saw_blep_osc_core.py**
➡ **ddsp_square_blep_osc_core.py**
➡ **ddsp_pulse_blep_osc_core.py**
➡ **ddsp_fm_osc_core.py**
➡ **ddsp_wavetable_multi_osc_core.py**
➡ **ddsp_grain_osc_core.py**

Or simply:

**“Generate all remaining oscillator modules.”**
