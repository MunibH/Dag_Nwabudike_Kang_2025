"""
wavelet_utils.py
================
Wavelet-based time–frequency analysis, causal filtering, cycle-phase
detection, and supporting signal-processing utilities.

All smoothing / filtering helpers are **causal** (output at time *t*
depends only on samples at times ≤ *t*) unless explicitly noted.

References
----------
* Morlet wavelet parameterisation: Katz et al., Nature Comms 2019.
* Group Sparse Total Variation: Selesnick & Chen, IEEE Sig. Proc. Lett. 2013.
"""

import numpy as np
from scipy import signal
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1.  MORLET WAVELET SPECTRUM                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

def morlet_wavelet(t, f, sigma):
    r"""
    Complex Morlet wavelet:

    .. math::
        \psi(t,f) = (\sigma\sqrt{\pi})^{-1/2}
                     \exp\!\bigl(-t^2/(2\sigma^2)\bigr)
                     \exp(2\pi i f t)

    Parameters
    ----------
    t : array_like
        Time array (seconds).
    f : float
        Centre frequency (Hz).
    sigma : float
        Gaussian envelope width (seconds).

    Returns
    -------
    psi : ndarray (complex)
    """
    norm = (sigma * np.sqrt(np.pi)) ** (-0.5)
    return norm * np.exp(-t**2 / (2 * sigma**2)) * np.exp(2j * np.pi * f * t)


def get_gaussian_sigma(f, ksi=4):
    r"""
    Gaussian envelope width σ = ξ / (2πf).

    Default ξ = 4 (Katz et al., Nature Comms 2019).
    """
    return ksi / (2 * np.pi * f)


def morlet_wavelet_ft(f, f0, sigma):
    r"""
    Analytical Fourier transform of the Morlet wavelet.

    .. math::
        \hat\Psi(f) = (\sigma\sqrt\pi)^{-1/2}\,\sigma\sqrt{2\pi}\,
                      \exp\!\bigl(-2\pi^2\sigma^2(f-f_0)^2\bigr)
    """
    norm = (sigma * np.sqrt(np.pi)) ** (-0.5)
    return norm * sigma * np.sqrt(2 * np.pi) * np.exp(
        -2 * np.pi**2 * sigma**2 * (f - f0) ** 2
    )


def pad_reflect(sig):
    """
    Pad a 1-D signal with reflected copies on both ends.

    Returns
    -------
    padded : ndarray, shape (3n − 2,)
    n_pad  : int  – samples added on the left.
    """
    left = sig[::-1][:-1]
    right = sig[::-1][1:]
    return np.concatenate([left, sig, right]), len(left)


def wavelet_spectrum(sig, freqs, dt, ksi=4):
    r"""
    Complex wavelet spectrum W(t, f) using Morlet wavelets.

    Computed in the frequency domain with reflected-edge padding.

    Parameters
    ----------
    sig   : 1-D or (n, 1) array – input time-series.
    freqs : 1-D array – analysis frequencies (Hz).
    dt    : float – sampling interval (s).
    ksi   : float – wavelet parameter (default 4).

    Returns
    -------
    W : ndarray, shape (n_time, n_freq), complex
    """
    if sig.ndim == 2 and sig.shape[1] == 1:
        sig = sig[:, 0]
    n_orig = len(sig)

    padded, n_pad = pad_reflect(sig)
    n_padded = len(padded)
    S_fft = np.fft.fft(padded)
    freqs_fft = np.fft.fftfreq(n_padded, d=dt)

    W = np.zeros((n_orig, len(freqs)), dtype=complex)
    for i, f0 in enumerate(freqs):
        sigma = get_gaussian_sigma(f0, ksi)
        Psi_hat = morlet_wavelet_ft(freqs_fft, f0, sigma)
        W_full = np.fft.ifft(S_fft * Psi_hat)
        W[:, i] = W_full[n_pad: n_pad + n_orig]
    return W


def time_averaged_spectrum(W, start_idx=None, end_idx=None):
    """Time-average the wavelet amplitude |W(t, f)|."""
    return np.mean(np.abs(W[start_idx:end_idx, :]), axis=0)


def weighted_average_spectra(spectra, weights):
    """Weighted average of time-averaged spectra across animals."""
    spectra = np.array(spectra)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return np.average(spectra, axis=0, weights=weights)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2.  NORMALISATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

def minmax_normalize(trace):
    """Scale *trace* to [0, 1].  NaN / Inf safe."""
    trace = np.asarray(trace, dtype=float)
    # Replace non-finite values with 0 so min/max don't become NaN
    clean = np.where(np.isfinite(trace), trace, 0.0)
    mn, mx = np.nanmin(clean), np.nanmax(clean)
    if mx == mn:
        return np.zeros_like(trace, dtype=float)
    out = (clean - mn) / (mx - mn)
    return out


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3.  GROUP SPARSE TOTAL VARIATION (GSTV) DENOISING                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def gstv(y, k, lam, n_outer=15, n_inner=30, rho=1.0, eps=1e-10, tol=1e-6):
    r"""
    GSTV denoising (Selesnick & Chen 2013).

    Solves  argmin_x  ½‖y − x‖² + λ Σ_i ‖d_{i:i+k}‖₂
    via MM + ADMM.

    .. note::
       This is inherently **non-causal** (batch optimisation over the
       whole signal).  It is used only as a *pre-processing* step; all
       subsequent filtering is causal.
    """
    y = np.asarray(y, dtype=float)
    # Guard against NaN / Inf – replace with 0 so solve_banded won't crash
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    n = len(y)
    if n < 3:
        return y.copy()
    x = y.copy()
    nd = n - 1

    ab = np.empty((3, n))
    ab[0, 0] = 0.0
    ab[0, 1:] = -rho
    ab[1, :] = 1.0 + 2.0 * rho
    ab[1, 0] = 1.0 + rho
    ab[1, -1] = 1.0 + rho
    ab[2, :-1] = -rho
    ab[2, -1] = 0.0

    for _ in range(n_outer):
        x_prev = x.copy()
        dx = np.diff(x)
        dx2 = dx ** 2

        n_groups = max(1, nd - k + 1)
        if k >= nd:
            gnorms = np.array([np.sqrt(np.sum(dx2) + eps)])
        else:
            cs = np.empty(nd + 1)
            cs[0] = 0.0
            np.cumsum(dx2, out=cs[1:])
            gnorms = np.sqrt(cs[k: k + n_groups] - cs[:n_groups] + eps)

        inv_g = 1.0 / gnorms
        weights = np.convolve(inv_g, np.ones(min(k, nd)), mode='full')[:nd]

        z = np.zeros(nd)
        u = np.zeros(nd)
        thresh = lam * weights / rho

        for __ in range(n_inner):
            v = z - u
            Dtv = np.empty(n)
            Dtv[0] = -v[0]
            Dtv[1:-1] = v[:-1] - v[1:]
            Dtv[-1] = v[-1]
            x = solve_banded((1, 1), ab, y + rho * Dtv)

            Dx = np.diff(x)
            s = Dx + u
            z = np.sign(s) * np.maximum(np.abs(s) - thresh, 0.0)
            u += Dx - z

        if np.linalg.norm(x - x_prev) / (1.0 + np.linalg.norm(x_prev)) < tol:
            break
    return x


def smooth_neural_data_gstv(traces_array, k=100, lam=0.025, fluc_thresh_std=3):
    """
    GSTV smoothing with extreme-fluctuation interpolation.

    Parameters
    ----------
    traces_array : (n_time, n_neurons) array
    k, lam       : GSTV parameters.
    fluc_thresh_std : jump threshold in std units.

    Returns
    -------
    smooth_traces, interpolated_traces : same shape as input.
    """
    traces = traces_array.copy().astype(float)
    # Replace any NaN / Inf with 0 before processing
    traces = np.where(np.isfinite(traces), traces, 0.0)
    n_time, n_neurons = traces.shape
    smooth = np.zeros_like(traces)
    interp = np.zeros_like(traces)

    for n in range(n_neurons):
        tr = traces[:, n]
        thr = fluc_thresh_std * np.std(tr)
        for i in range(1, n_time - 1):
            if abs(tr[i] - tr[i - 1]) > thr:
                tr[i] = (tr[i + 1] + tr[i - 1]) / 2.0
        interp[:, n] = gstv(tr, 100, 0.01)
        smooth[:, n] = gstv(tr, k, lam)
        if (n + 1) % 10 == 0 or n == n_neurons - 1:
            print(f'  neuron {n + 1}/{n_neurons}', end='\r')
    print()
    return smooth, interp


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4.  CAUSAL FILTERING                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# Key design decisions
# --------------------
# * `filtfilt` is **zero-phase / non-causal** → replaced by `sosfilt`
#   (forward-only IIR) which is strictly causal.
# * `gaussian_filter1d` is symmetric / non-causal → replaced by a causal
#   exponential moving average (1st-order IIR low-pass) with matched
#   effective σ.
# * All public filter functions below use second-order-section (SOS)
#   representations for numerical stability.

def _sos_butter(order, Wn, btype, fs):
    """Design a Butterworth SOS filter at the given *fs* (Hz)."""
    return signal.butter(order, Wn, btype=btype, fs=fs, output='sos')


def causal_lowpass(data, cutoff_hz, fs, order=4):
    """
    Forward-only Butterworth low-pass (causal).

    Parameters
    ----------
    data      : 1-D array
    cutoff_hz : −3 dB cutoff frequency (Hz).
    fs        : sampling rate (Hz).
    order     : filter order (default 4).

    Returns
    -------
    filtered : 1-D array  (same length; has phase lag).
    """
    sos = _sos_butter(order, cutoff_hz, 'low', fs)
    return signal.sosfilt(sos, data)


def causal_highpass(data, cutoff_hz, fs, order=4):
    """
    Forward-only Butterworth high-pass (causal).
    """
    sos = _sos_butter(order, cutoff_hz, 'high', fs)
    return signal.sosfilt(sos, data)


def causal_bandpass(data, low_hz, high_hz, fs, order=4):
    """
    Forward-only Butterworth band-pass (causal).

    Parameters
    ----------
    data    : 1-D array
    low_hz  : lower −3 dB frequency (Hz).
    high_hz : upper −3 dB frequency (Hz).
    fs      : sampling rate (Hz).
    order   : filter order per band edge (default 4).

    Returns
    -------
    filtered : 1-D array
    """
    sos = _sos_butter(order, [low_hz, high_hz], 'band', fs)
    return signal.sosfilt(sos, data)


def causal_ema(data, sigma_samples):
    """
    Causal exponential-moving-average low-pass.

    The decay constant α is chosen so the effective standard deviation
    of the impulse response matches *sigma_samples*, giving behaviour
    comparable to ``gaussian_filter1d`` but strictly causal.

    Parameters
    ----------
    data          : 1-D array
    sigma_samples : effective width in samples (analogous to σ for a
                    Gaussian kernel).

    Returns
    -------
    smoothed : 1-D array
    """
    # For an EMA with parameter α, the std of the (one-sided)
    # exponential impulse response is  (1 − α) / α.
    # Setting that equal to sigma_samples:  α = 1 / (sigma_samples + 1)
    alpha = 1.0 / (sigma_samples + 1.0)
    b = np.array([alpha])
    a = np.array([1.0, -(1.0 - alpha)])
    return signal.lfilter(b, a, data)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5.  RIGOROUS CUTOFF SELECTION                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def estimate_psd(data, fs, nperseg=None, noverlap=None):
    """
    Welch PSD estimate.

    Returns
    -------
    f_psd   : frequency axis (Hz).
    psd     : power spectral density.
    """
    if nperseg is None:
        nperseg = min(len(data), int(fs * 120))   # ~2-min windows
    if noverlap is None:
        noverlap = nperseg // 2
    return signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)


def dominant_frequency_band(f_psd, psd, noise_floor_quantile=0.25,
                            threshold_db=-6):
    """
    Identify the dominant frequency band from a PSD.

    Algorithm
    ---------
    1. Find the frequency with maximum power (*f_peak*).
    2. Estimate the noise floor as the *noise_floor_quantile* quantile
       of the PSD.
    3. Define the band edges as the frequencies where the PSD drops to
       *threshold_db* dB below the peak (relative to the noise floor).

    Parameters
    ----------
    f_psd  : 1-D array – frequency axis (Hz).
    psd    : 1-D array – power spectral density.
    noise_floor_quantile : float – quantile for noise floor estimate.
    threshold_db : float – dB below peak for band edges (default −6 dB,
                   i.e. quarter-power).

    Returns
    -------
    f_peak   : float – peak frequency (Hz).
    f_low    : float – lower band edge (Hz).
    f_high   : float – upper band edge (Hz).
    snr_db   : float – peak SNR in dB above noise floor.
    """
    psd = np.asarray(psd, dtype=float)
    noise_floor = np.quantile(psd, noise_floor_quantile)

    # Peak
    i_peak = np.argmax(psd)
    f_peak = f_psd[i_peak]
    peak_power = psd[i_peak]

    snr_db = 10.0 * np.log10(peak_power / max(noise_floor, 1e-30))

    # Threshold in linear scale
    thresh_lin = peak_power * 10.0 ** (threshold_db / 10.0)

    # Walk left from peak to find f_low
    i_low = i_peak
    while i_low > 0 and psd[i_low] > thresh_lin:
        i_low -= 1
    f_low = max(f_psd[i_low], f_psd[1])  # avoid DC

    # Walk right from peak to find f_high
    i_high = i_peak
    while i_high < len(psd) - 1 and psd[i_high] > thresh_lin:
        i_high += 1
    f_high = f_psd[i_high]

    return f_peak, f_low, f_high, snr_db


def auto_bandpass_cutoffs(data, fs, nperseg=None, noise_floor_quantile=0.25,
                          threshold_db=-6, margin_factor=1.5):
    """
    Automatically determine bandpass cutoffs from the data's PSD.

    Computes the PSD, finds the dominant oscillation band via
    :func:`dominant_frequency_band`, then widens the band by
    *margin_factor* on each side (in log-frequency space) to ensure the
    filter does not attenuate the signal of interest.

    Parameters
    ----------
    data   : 1-D array.
    fs     : sampling rate (Hz).
    nperseg : int or None – Welch segment length.
    noise_floor_quantile : float.
    threshold_db : float – dB threshold for band detection.
    margin_factor : float – multiplicative margin applied to the band
                    edges (default 1.5×).

    Returns
    -------
    low_hz  : float – recommended lower cutoff.
    high_hz : float – recommended upper cutoff.
    info    : dict  – diagnostic information including PSD, peak
              frequency, SNR, and raw band edges.
    """
    f_psd, psd = estimate_psd(data, fs, nperseg=nperseg)
    f_peak, f_low, f_high, snr_db = dominant_frequency_band(
        f_psd, psd, noise_floor_quantile, threshold_db
    )

    # Apply margin (in log-frequency space for symmetry on a log axis)
    low_hz = f_low / margin_factor
    high_hz = f_high * margin_factor

    # Clamp to valid filter range
    nyq = fs / 2.0
    low_hz = max(low_hz, f_psd[1])       # avoid 0 Hz
    high_hz = min(high_hz, 0.95 * nyq)   # stay below Nyquist

    info = dict(
        f_psd=f_psd, psd=psd,
        f_peak=f_peak,
        f_low_raw=f_low, f_high_raw=f_high,
        snr_db=snr_db,
        margin_factor=margin_factor,
    )
    return low_hz, high_hz, info


def plot_psd_with_cutoffs(info, low_hz, high_hz, ax=None):
    """
    Diagnostic plot: PSD with detected band and chosen cutoffs.

    Parameters
    ----------
    info     : dict returned by :func:`auto_bandpass_cutoffs`.
    low_hz, high_hz : chosen cutoffs (Hz).
    ax       : matplotlib Axes (created if *None*).

    Returns
    -------
    ax : matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    f, psd = info['f_psd'], info['psd']
    ax.semilogy(f, psd, 'k', lw=1)
    ax.axvline(info['f_peak'], color='red', ls='--', lw=1,
               label=f'peak = {info["f_peak"]:.4f} Hz')
    ax.axvspan(info['f_low_raw'], info['f_high_raw'],
               alpha=0.15, color='blue', label='detected band')
    ax.axvline(low_hz, color='green', ls=':', lw=1.2,
               label=f'BP low = {low_hz:.4f} Hz')
    ax.axvline(high_hz, color='green', ls=':', lw=1.2,
               label=f'BP high = {high_hz:.4f} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(f'PSD  |  peak SNR = {info["snr_db"]:.1f} dB')
    ax.legend(fontsize=8, loc='upper right')
    return ax


def plot_filter_response(sos, fs, n_freqs=2048, ax=None):
    """
    Plot the magnitude response of a SOS filter.

    Parameters
    ----------
    sos : SOS array from scipy.signal.butter(..., output='sos').
    fs  : sampling rate (Hz).
    ax  : matplotlib Axes (created if None).

    Returns
    -------
    ax
    """
    w, h = signal.sosfreqz(sos, worN=n_freqs, fs=fs)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    ax.plot(w, 20 * np.log10(np.abs(h) + 1e-30), 'k', lw=1.2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title('Filter magnitude response')
    ax.set_ylim(bottom=-80)
    ax.axhline(-3, color='grey', ls='--', lw=0.7, label='−3 dB')
    ax.legend(fontsize=8)
    return ax


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6.  NORMALISE → CAUSAL SMOOTH → DERIVATIVE  PIPELINE              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def normalize_smooth_derivative(trace, dt, sigma_seconds=1.0, causal=True):
    """
    Min-max normalise → low-pass smooth → first derivative.

    Parameters
    ----------
    trace         : 1-D array
    dt            : sampling interval (s).
    sigma_seconds : smoothing width (s).
    causal        : if True (default) use causal EMA; else Gaussian.

    Returns
    -------
    normalized, smoothed, derivative : 1-D arrays.
    """
    trace = np.asarray(trace, dtype=float)
    # Sanitise non-finite values before any processing
    trace = np.where(np.isfinite(trace), trace, 0.0)
    normalized = minmax_normalize(trace)
    sigma_samples = sigma_seconds / dt
    if causal:
        smoothed = causal_ema(normalized, sigma_samples)
    else:
        smoothed = gaussian_filter1d(normalized, sigma=sigma_samples)
    derivative = np.gradient(smoothed, dt)
    return normalized, smoothed, derivative


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7.  CYCLE-PHASE DETECTION                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def find_oscillation_cycles(trace, fs, min_peak_distance_sec=5.0,
                            peak_prominence=None):
    """
    Detect oscillation cycles and label rising / falling phases.

    Algorithm
    ---------
    1. Find local maxima (peaks) and minima (troughs) in *trace*.
    2. Pair successive trough → peak (rising) and peak → trough (falling)
       segments.
    3. Return arrays of cycle boundaries and a per-sample phase label.

    Parameters
    ----------
    trace : 1-D array – (ideally band-pass filtered) signal.
    fs    : float – sampling rate (Hz).
    min_peak_distance_sec : float – minimum time between successive
        peaks (seconds). Set to ~ half the expected period.
    peak_prominence : float or None – minimum prominence for peak
        detection. If None, uses 0.1 × std(trace).

    Returns
    -------
    cycles : dict with keys
        'peaks'    : indices of detected peaks.
        'troughs'  : indices of detected troughs.
        'rising'   : list of (start, end) index pairs.
        'falling'  : list of (start, end) index pairs.
        'phase'    : ndarray, same length as *trace*:
                     +1 = rising, −1 = falling, 0 = unclassified.
        'inst_phase' : ndarray – Hilbert instantaneous phase (radians).
    """
    min_dist = int(min_peak_distance_sec * fs)
    if peak_prominence is None:
        peak_prominence = 0.1 * np.std(trace)

    peaks, _ = signal.find_peaks(trace, distance=min_dist,
                                 prominence=peak_prominence)
    troughs, _ = signal.find_peaks(-trace, distance=min_dist,
                                   prominence=peak_prominence)

    # --- pair troughs and peaks into rising / falling segments ---
    rising, falling = [], []

    # Merge and sort all extrema with labels
    extrema = ([(p, 'peak') for p in peaks] +
               [(t, 'trough') for t in troughs])
    extrema.sort(key=lambda x: x[0])

    for j in range(len(extrema) - 1):
        idx_a, label_a = extrema[j]
        idx_b, label_b = extrema[j + 1]
        if label_a == 'trough' and label_b == 'peak':
            rising.append((idx_a, idx_b))
        elif label_a == 'peak' and label_b == 'trough':
            falling.append((idx_a, idx_b))

    # Per-sample phase label
    phase = np.zeros(len(trace), dtype=int)
    for a, b in rising:
        phase[a:b + 1] = 1
    for a, b in falling:
        phase[a:b + 1] = -1

    # Hilbert instantaneous phase (for continuous phase readout)
    analytic = signal.hilbert(trace - np.mean(trace))
    inst_phase = np.angle(analytic)

    return dict(
        peaks=peaks,
        troughs=troughs,
        rising=rising,
        falling=falling,
        phase=phase,
        inst_phase=inst_phase,
    )


def plot_cycles(time_axis, trace, cycles, ax=None, title='',
                rising_color='#2ca02c', falling_color='#d62728',
                peak_color='red', trough_color='blue'):
    """
    Plot a trace with rising / falling phases shaded.

    Parameters
    ----------
    time_axis : 1-D array (seconds).
    trace     : 1-D array.
    cycles    : dict from :func:`find_oscillation_cycles`.
    ax        : matplotlib Axes (created if None).
    title     : str.

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    ax.plot(time_axis, trace, 'k', lw=0.8)

    for a, b in cycles['rising']:
        ax.axvspan(time_axis[a], time_axis[b],
                   alpha=0.20, color=rising_color, label=None)
    for a, b in cycles['falling']:
        ax.axvspan(time_axis[a], time_axis[b],
                   alpha=0.20, color=falling_color, label=None)

    ax.plot(time_axis[cycles['peaks']], trace[cycles['peaks']],
            'v', color=peak_color, ms=6, label='peak')
    ax.plot(time_axis[cycles['troughs']], trace[cycles['troughs']],
            '^', color=trough_color, ms=6, label='trough')

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    # Add shading legend entries manually
    from matplotlib.patches import Patch
    handles += [Patch(facecolor=rising_color, alpha=0.3, label='rising'),
                Patch(facecolor=falling_color, alpha=0.3, label='falling')]
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    if title:
        ax.set_title(title)
    return ax


def plot_instantaneous_phase(time_axis, inst_phase, ax=None, title=''):
    """
    Plot Hilbert instantaneous phase over time.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(time_axis, inst_phase, 'k', lw=0.6)
    ax.set_ylabel('Phase (rad)')
    ax.set_xlabel('Time (s)')
    if title:
        ax.set_title(title)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    return ax


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  8.  CONVENIENCE / PLOTTING HELPERS                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Label → colour mapping for g5HT ROI data
LABEL_COLORS = {
    'procorpus': '#F94144', 'PC': '#F94144',
    'metacorpus': '#FF9129', 'MC': '#FF9129',
    'isthmus': '#F3BD3E', 'IM': '#F3BD3E',
    'terminal_bulb': '#FFD981', 'TB': '#FFD981',
    'nerve_ring': '#90BE6D', 'NR': '#90BE6D',
    'ventral_nerve_cord': '#43AA8B', 'VNC': '#43AA8B',
    'dorsal_nerve_cord': '#000000', 'DNC': '#000000',
}
_FALLBACK = ['#17becf', '#bcbd22', '#7f7f7f', '#aec7e8', '#ffbb78', '#98df8a']


def get_label_color(label, fallback_idx=0):
    """Return the hex colour for a given ROI label."""
    return LABEL_COLORS.get(label, _FALLBACK[fallback_idx % len(_FALLBACK)])


def get_nsm_trace(data, trace_key='trace_array'):
    """
    Extract the NSM trace from an HDF5-derived data dict.

    If multiple NSM indices exist, returns their average.
    """
    nsm_idx = data['gcamp']['idx_nsm']
    arr = data['gcamp'][trace_key]
    if hasattr(nsm_idx, '__len__') and len(nsm_idx) > 1:
        cols = np.column_stack([arr[:, int(i) - 1] for i in nsm_idx])
        return np.mean(cols, axis=1)
    return arr[:, int(nsm_idx) - 1]


def compute_and_plot_wavelet(trace, freqs, dt, ksi=4.0,
                             food_idx=None, cmap='YlOrBr',
                             pretty_subplots_fn=None):
    """
    One-call helper: compute wavelet spectrum, plot spectrogram and
    time-averaged spectrum.

    Parameters
    ----------
    trace   : 1-D array (pre-processed signal).
    freqs   : frequency axis for wavelet analysis (Hz).
    dt      : sampling interval (s).
    ksi     : wavelet parameter.
    food_idx : int or None – frame index for food encounter line.
    cmap    : colourmap name.
    pretty_subplots_fn : optional callable for styled subplots.

    Returns
    -------
    W           : complex wavelet spectrum.
    avg_spectrum : time-averaged amplitude spectrum.
    """
    W = wavelet_spectrum(trace, freqs, dt, ksi=ksi)
    time_axis = np.arange(len(trace)) * dt

    # --- Spectrogram ---
    if pretty_subplots_fn is not None:
        fig, axes, nt = pretty_subplots_fn(nrows=2, ncols=1,
                                           figsize=(9, 4), sharex=True)
        ax0, ax1 = nt(), nt()
    else:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4), sharex=True)

    ax0.plot(time_axis, trace, 'k', lw=0.8)
    if food_idx is not None:
        ax0.axvline(food_idx * dt, color='black', ls='--')
        ax1.axvline(food_idx * dt, color='black', ls='--')
    ax0.set_ylabel('Signal')

    im = ax1.pcolormesh(time_axis, freqs, np.abs(W).T,
                        shading='auto', cmap=cmap)
    ax1.set_ylabel('Freq (Hz)')
    ax1.set_xlabel('Time (s)')
    fig.colorbar(im, ax=ax1, label='Amp. (a.u.)')
    plt.show()

    # --- Time-averaged spectrum ---
    start = int(food_idx) if food_idx is not None else None
    avg_spectrum = time_averaged_spectrum(W, start_idx=start)

    fig2, ax2 = plt.subplots(figsize=(2.4, 4.5))
    ax2.plot(avg_spectrum, freqs, 'k', lw=1.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Amp. (a.u.)')
    ax2.set_ylabel('Freq (Hz)')
    plt.tight_layout()
    plt.show()

    return W, avg_spectrum
