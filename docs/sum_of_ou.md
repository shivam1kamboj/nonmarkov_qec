# Sum-of-OU: approximating 1/f noise

This note records the design for `SumOfOUProcess`, the multi-component 
noise generator that approximates a 1/f spectrum by summing independent 
Ornstein-Uhlenbeck processes. It is the analytic specification; 
implementation follows in `src/nonmarkov_qec/noise/`.

## Motivation

A single OU process has a Lorentzian power spectrum:

    S_OU(f) = 2 sigma^2 tau_c / (1 + (2 pi f tau_c)^2)

which is flat for f << 1/(2 pi tau_c) and falls as f^-2 above. Real 
superconducting
qubits exhibit 1/f noise — power spectral density scaling approximately as 
1/f over
several decades of frequency, dominated by low-frequency flux and charge 
fluctuations.

A 1/f spectrum cannot come from a single OU process, but it can be 
approximated by a
**sum of OU processes** with logarithmically-spaced correlation times. 
Each component
contributes a Lorentzian; stacked with appropriate spacing and weights, 
the sum tiles
the frequency axis and approximates 1/f over a target band.

## Construction

Define the summed process as

    X_sum(t) = sum_{j=1}^{N} X_j(t),    X_j ~ OU(tau_j, sigma_j)

with the components mutually independent (independent driving noise, 
independent
initial conditions).

### Correlation times: logarithmic spacing

Place N correlation times log-spaced between tau_min and tau_max:

    tau_j = tau_min * (tau_max / tau_min)^((j-1)/(N-1)),   j = 1, ..., N

The corner frequency of component j is f_j = 1 / (2 pi tau_j). Log-spacing 
the tau_j
log-spaces the corner frequencies, so the individual Lorentzian "knees" 
are spread
evenly across the frequency axis on a log scale. This is what produces an 
approximately
1/f sum.

### Weights: equal per-component variance (v1)

For v1 we use equal variance per component:

    sigma_j^2 = sigma_total^2 / N    for all j

so that the total stationary variance is

    Var(X_sum) = sum_j sigma_j^2 = sigma_total^2

(using independence of the components).

**Why equal weights gives approximately 1/f.** With log-spaced tau_j, each 
Lorentzian
dominates roughly one logarithmic frequency interval near its corner f_j. 
Equal-variance
components contribute equal power per logarithmic frequency interval, 
which is exactly
the defining property of a 1/f spectrum (constant power per 
octave/decade). The
approximation improves with more components per decade. See Kogan, 
*Electronic Noise
and Fluctuations in Solids*, for the standard treatment.

This is the simplest defensible weighting. A refined weighting that 
targets a precise
1/f^alpha slope is possible (adjusting sigma_j^2 with tau_j) and is 
deferred to a
later version if the equal-weight slope proves insufficient.

## Analytic spectrum

By independence, the summed spectrum is the sum of the component spectra:

    S_sum(f) = sum_{j=1}^{N} 2 sigma_j^2 tau_j / (1 + (2 pi f tau_j)^2)

The summed autocorrelation is likewise the sum of the component 
autocorrelations:

    C_sum(tau) = sum_{j=1}^{N} sigma_j^2 exp(-|tau| / tau_j)

Both follow directly from the independence of the components and the known 
single-OU
results. These are the analytic references the validation tests will check 
against.

## Target band and the 1/f region

The summed spectrum approximates 1/f only in the band between the lowest 
and highest
corner frequencies:

    f_low_corner  = 1 / (2 pi tau_max)     (set by the longest correlation 
time)
    f_high_corner = 1 / (2 pi tau_min)     (set by the shortest 
correlation time)

- Below f_low_corner: the spectrum flattens (all components are in their 
plateau).
- Above f_high_corner: the spectrum rolls off as f^-2 (all components past 
their knee).
- Between: approximately 1/f.

So tau_max controls the low-frequency edge of the 1/f region and tau_min 
controls the
high-frequency edge. A rule of thumb of >= 2-3 components per decade of 
frequency keeps
the 1/f approximation within a few percent across the band.

## API

### Core constructor (units: correlation time)

    SumOfOUProcess(tau_min, tau_max, n_components, sigma_total)

The physics lives in correlation times, so this is the primitive 
interface. Validation
tests check against this.

### Convenience wrapper (units: frequency)

    SumOfOUProcess.from_frequency_band(f_min, f_max, n_components, 
sigma_total)

Users often think in terms of "1/f noise between f_min and f_max." The 
corner-frequency
relation f = 1/(2 pi tau) inverts the band to a tau range:

    tau_min = 1 / (2 pi f_max)     (high frequency  -> short tau)
    tau_max = 1 / (2 pi f_min)     (low frequency   -> long tau)

The wrapper performs this conversion and delegates to the core 
constructor. Note the
inversion: the high-frequency edge of the band maps to the *shortest* 
correlation time.

### sample()

Same signature and output contract as OUProcess.sample():

    sample(n_steps, dt, n_trajectories=1, rng=None, x0=None)
        -> ndarray of shape (n_trajectories, n_steps + 1)

Implementation: sample each component independently (reusing OUProcess 
internally or
the same exact-update recursion), then sum across components. A single rng 
is threaded
through so the whole sum is reproducible from one seed.

## Validation plan

The single-OU validation framework carries over; only the analytic 
references change.

1. **Stationary distribution.** X_sum ~ N(0, sigma_total^2). Same test as 
single-OU
   with the summed variance.
2. **Autocorrelation.** Empirical C(tau) matches the sum-of-exponentials
   C_sum(tau) above.
3. **Power spectrum.** Empirical Welch PSD matches the sum-of-Lorentzians 
S_sum(f).
   Note: as with single-OU, the correct reference for the 
discretely-sampled process is
   the sum of *discrete-time* AR(1) spectra, not the continuous 
sum-of-Lorentzians —
   the two differ near Nyquist. The discrete reference is the sum over 
components of
   the single-OU discrete spectrum.
4. **1/f slope.** New test specific to the sum: fit the log-log slope of 
the empirical
   (or analytic) spectrum over the target band [f_low_corner, 
f_high_corner] and check
   it is close to -1. Tolerance to be set during implementation; expect 
the slope to be
   within ~0.1 of -1 for >= 3 components per decade.

## Open questions (deferred)

- **Refined weighting for precise 1/f^alpha.** Equal weights give 
approximately 1/f;
  a tau-dependent weighting can target a specific slope. Defer unless the 
equal-weight
  slope test fails.
- **Number of components per decade.** Default and validation tolerance to 
be chosen
  during implementation, guided by the slope test.
- **Whether to expose the per-component tau_j and sigma_j as read-only 
attributes** for
  inspection/plotting. Likely yes; decide during implementation.
