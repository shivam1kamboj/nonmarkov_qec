# Noise generator: math, validation, and API

This document describes the `OUProcess` sampler in 
`src/nonmarkov_qec/noise/ornstein_uhlenbeck.py`, the math behind it, and 
the four statistical tests that validate it against analytic predictions.

## The Ornstein-Uhlenbeck process

The Ornstein-Uhlenbeck (OU) process is the simplest stationary Gaussian 
process with memory. It satisfies the stochastic differential equation

$$
dX = -\frac{X}{\tau_c}\,dt + \sqrt{\frac{2\sigma^2}{\tau_c}}\,dW
$$

with two parameters:

- $\tau_c > 0$ — the correlation time, setting how long the process 
"remembers" its past value.
- $\sigma > 0$ — the stationary standard deviation, setting the amplitude.

Three properties follow from the SDE and are used as test targets:

**Stationary distribution.** Once transients have decayed, $X \sim 
\mathcal{N}(0, \sigma^2)$.

**Autocorrelation function.**

$$
\langle X(t)\,X(t+\tau)\rangle = \sigma^2\,e^{-|\tau|/\tau_c}.
$$

Memory decays exponentially with timescale $\tau_c$. This is the defining 
feature of OU — the process is non-Markovian on timescales shorter than 
$\tau_c$ and effectively Markovian on timescales much longer than 
$\tau_c$.

**Power spectrum.** By the Wiener-Khinchin theorem, the spectrum is the 
Fourier transform of the autocorrelation:

$$
S(f) = \frac{2\sigma^2\,\tau_c}{1 + (2\pi f \tau_c)^2}.
$$

This is a Lorentzian. Flat for $f \ll 1/(2\pi\tau_c)$, $f^{-2}$ rolloff 
above.

## Why this matters for the project

In the context of QEC, we treat $X(t)$ as a stochastic fluctuation in a 
qubit's energy (e.g., flux noise contributing to dephasing). The relevant 
question is how the temporal correlations in $X(t)$ — characterized by 
$\tau_c$ — affect the threshold of error-correcting codes that, by 
construction, assume time-independent error rates. A single OU process is 
the simplest non-trivial test bed: it has memory, it has a clean spectrum, 
and its Markovian limit ($\tau_c \to 0$ at fixed $D = \sigma^2 \tau_c$) 
recovers ordinary white noise.

For 1/f-like spectra observed in real hardware, the sampler will 
eventually be extended to a sum of OU processes with 
logarithmically-spaced $\tau_c$ values (see "Roadmap" below).

## The exact-update sampler

The OU SDE is linear with Gaussian noise, so it admits an analytic 
solution over any finite time interval $\Delta t$. Multiplying through by 
the integrating factor $e^{t/\tau_c}$ and integrating from $t$ to $t + 
\Delta t$ yields the closed-form update

$$
X_{k+1} = X_k\,e^{-\Delta t/\tau_c} + \sigma\sqrt{1 - e^{-2\Delta 
t/\tau_c}}\,Z_k,\quad Z_k \sim \mathcal{N}(0, 1).
$$

Each $Z_k$ is drawn independently. This recursion is *statistically 
identical* to the continuous-time SDE sampled at intervals $\Delta t$ — 
not approximate, not subject to an integration tolerance, and 
unconditionally stable. The exact-update form holds for any $\Delta t$, 
which is important because our downstream simulations use $\Delta t \sim 
\tau_c$ (one gate cycle).

Reference: Gillespie, "Exact numerical simulation of the 
Ornstein-Uhlenbeck process and its integral," Phys. Rev. E 54, 2084 
(1996).

### Implementation choices

Five small choices that matter:

1. **Precompute coefficients outside the loop.** $a = e^{-\Delta 
t/\tau_c}$ and the noise scale $\sigma\sqrt{1 - e^{-2\Delta t/\tau_c}}$ 
are constants of integration; computing them once instead of per step is a 
free speedup.

2. **Pre-generate the entire Gaussian noise batch.** 
`rng.standard_normal((n_trajectories, n_steps))` in one call is much 
faster than per-step draws inside a Python loop, and produces identical 
statistics.

3. **Vectorize across trajectories.** The Python loop is over time only; 
each step updates all trajectories simultaneously via numpy broadcasting. 
For typical runs ($10^4$ trajectories of $10^3$ steps), this is what makes 
the sampler tractable.

4. **Default initial condition is drawn from the stationary 
distribution.** When `x0=None`, $X_0 \sim \mathcal{N}(0, \sigma^2)$. This 
eliminates burn-in: the process is statistically stationary from step 0.

5. **RNG is an injected `numpy.random.Generator`.** Tests pass an explicit 
seed; downstream simulations construct generators from a seed parameter. 
No implicit global state.

The implementation lives in 
`src/nonmarkov_qec/noise/ornstein_uhlenbeck.py`.

## Validation tests

Four orthogonal statistical tests verify that the sampler produces the OU 
process it claims to. Each compares an empirical statistic computed from 
many sampled trajectories against an analytic prediction.

All tests live in `tests/test_ornstein_uhlenbeck.py` and use explicit 
seeds for deterministic reproducibility.

### Test 1: Stationary distribution

After burn-in, samples should be $\mathcal{N}(0, \sigma^2)$.

- $10{,}000$ trajectories, $2{,}000$ steps, $\Delta t = \tau_c/10$.
- Discard the first half of each trajectory as burn-in.
- Check $|\text{sample mean}| < 3\sigma/\sqrt{n_{\text{trajectories}}}$ 
(three standard errors).
- Check $|\text{sample variance} - \sigma^2|/\sigma^2 < 0.02$ (2% relative 
error).

### Test 2: Autocorrelation

Empirical $\hat{C}(\tau)$ should match $\sigma^2 e^{-|\tau|/\tau_c}$ 
within 5% for $\tau \in [0, 3\tau_c]$.

- Same sample configuration as Test 1.
- For each lag $k = 0, 1, \ldots, 30$ (in units of $\Delta t$), compute 
$\hat{C}(k) = \langle X_i(t)\,X_i(t + k\Delta t)\rangle$ averaged over 
trajectories $i$ and time $t$.
- Compare against the analytic exponential decay.

The 5% tolerance is loose enough to absorb Monte Carlo noise at the 
largest lags (where the signal is small) and tight enough to catch a wrong 
decay rate or wrong stationary variance.

### Test 3: Power spectrum

Empirical Welch PSD should match the *exact discrete-time AR(1) spectrum* 
within 5% across the resolved frequency band.

Because the sampler produces a discrete-time sequence, the appropriate 
reference is not the continuous Lorentzian $S(f) = 2\sigma^2\tau_c/(1 + 
(2\pi f \tau_c)^2)$ but the discrete-time formula

$$
S_{\text{discrete}}(f) = \frac{b^2\,\Delta t}{1 - 2a\cos(2\pi f \Delta t) 
+ a^2},
$$

with $a = e^{-\Delta t/\tau_c}$ and $b^2 = \sigma^2(1 - a^2)$. The 
continuous Lorentzian is only the $\Delta t \to 0$ limit of this 
expression; at finite $\Delta t$ the discrete spectrum differs from the 
Lorentzian — most prominently near the Nyquist frequency, where aliased 
power folds back.

- $2{,}000$ trajectories, $2{,}000$ steps, $\Delta t = \tau_c/10$.
- `scipy.signal.welch` with `nperseg = n_steps // 8`, vectorized across 
trajectories via `axis=-1`.
- The Welch PSD is one-sided by default (power doubled for $f > 0$ to 
preserve total variance under integration over $[0, f_{\text{Nyq}}]$). We 
multiply $S_{\text{discrete}}$ by 2 for $f > 0$ to match. The Nyquist bin 
is excluded from comparison because Welch does not double it.
- Comparison band: $[1/(10\tau_c), 1/(2\Delta t))$.

### Test 4: Markovian limit

This is the dynamical test. As $\tau_c \to 0$ at fixed $D = 
\sigma^2\tau_c$, the OU process should become indistinguishable from white 
noise with intensity $2D$. The signature is that the time-integral

$$
S(t) = \int_0^t X(s)\,ds
$$

should approach a Brownian motion with $\text{Var}(S(t)) \to 2Dt$ as 
$\tau_c \to 0$. The full finite-$\tau_c$ prediction, derived by direct 
integration of the autocorrelation, is

$$
\text{Var}\bigl(S(t)\bigr) = 2Dt - 2D\tau_c\bigl(1 - e^{-t/\tau_c}\bigr).
$$

The second term is the "memory correction" that vanishes when $\tau_c \ll 
t$.

- Fixed $D = 0.1$, fixed observation window $t = 5$ (absolute time units).
- $\tau_c \in \{1.0, 0.1, 0.01\}$ with independent seeds.
- For each $\tau_c$: sample $5{,}000$ trajectories on a grid of 
$n_{\text{window}} = t / \Delta t$ steps with $\Delta t = \tau_c/10$, 
drawing $X_0$ from the stationary distribution (no burn-in needed).
- Compute $S_i = \Delta t \sum_k X_{i,k}$ (left-endpoint Riemann sum), and 
check that the empirical $\text{Var}(S)$ matches the analytic formula 
within 5% relative error at every $\tau_c$.

At $\tau_c = 1$ the correction is ~20% of the leading term; at $\tau_c = 
0.01$ it is ~0.2%. Passing the test at all three values demonstrates that 
the sampler approaches the correct white-noise limit with the correct 
rate.

### Why these four

Each test catches a different class of bug.

| Test | Catches |
|---|---|
| Stationary | wrong amplitude (factor in noise scale), wrong long-time 
mean |
| Autocorrelation | wrong correlation time (wrong $a$ in the recursion), 
wrong stationary variance |
| Power spectrum | wrong discrete-time structure, normalization errors, 
aliasing bugs |
| Markovian limit | bugs in the recursion that distort the time-integral 
while preserving equilibrium moments |

Tests 2 and 3 nominally check the same physical content (two-time 
statistics in conjugate domains) but in practice catch different bugs: 
aliasing-edge issues show up sharply in test 3 but only weakly in test 2.

## API

```python
from nonmarkov_qec.noise.ornstein_uhlenbeck import OUProcess
import numpy as np

ou = OUProcess(tau_c=1.0, sigma=0.5)

rng = np.random.default_rng(seed=42)
trajectories = ou.sample(
    n_steps=1000,
    dt=0.1,
    n_trajectories=10_000,
    rng=rng,
)
# trajectories.shape == (10_000, 1001)
# trajectories[i, k] = X_i at time k * dt
# X_0 (column 0) drawn from the stationary distribution N(0, sigma^2)
```

Input validation:

- `n_steps` must be $\geq 0$.
- `dt` must be $> 0$.
- `tau_c` and `sigma` must be $> 0$ (enforced at construction).

Output:

- Shape `(n_trajectories, n_steps + 1)`. Index 0 is the initial state; 
subsequent indices are the sampled trajectory at time $k\Delta t$.
- Dtype `float64`.

## Roadmap

This document covers the single-OU sampler only. Future extensions:

- **Sum-of-OU framework** for 1/f-like spectra over a target frequency 
band. The single-OU validation infrastructure carries over directly; only 
the reference autocorrelation and spectrum change (sum of Lorentzians 
instead of one).
- **Validation plots.** Empirical-vs-analytic overlays for the four tests, 
generated programmatically and saved to `docs/figures/`. To be added.
