"""Pure signal-processing helpers (no UI code), so they are easy to test."""
import numpy as np
from scipy import signal

WAVEFORMS = ["Sine", "Square", "Sawtooth", "Triangle", "Two-tone (f + 2.5f)"]
FILTERS = ["None", "Low-pass", "High-pass", "Band-pass", "Band-stop"]


def generate(kind, freq, amp, fs, duration, noise_level=0.0, seed=None):
    """Return (t, x) for the chosen waveform with optional white noise."""
    t = np.arange(0, duration, 1.0 / fs)
    w = 2 * np.pi * freq * t
    if kind == "Sine":
        x = np.sin(w)
    elif kind == "Square":
        x = signal.square(w)
    elif kind == "Sawtooth":
        x = signal.sawtooth(w)
    elif kind == "Triangle":
        x = signal.sawtooth(w, width=0.5)
    elif kind == "Two-tone (f + 2.5f)":
        x = 0.7 * np.sin(w) + 0.5 * np.sin(2.5 * w)
    else:
        raise ValueError(f"Unknown waveform: {kind}")
    x = amp * x
    if noise_level > 0:
        rng = np.random.default_rng(seed)
        x = x + noise_level * rng.standard_normal(len(t))
    return t, x


def apply_filter(x, fs, kind, low, high=0.0, order=4):
    """Zero-phase Butterworth filter. Raises ValueError on invalid cutoffs."""
    if kind == "None":
        return x.copy()
    nyq = fs / 2.0
    if kind in ("Low-pass", "High-pass"):
        if not 0 < low < nyq:
            raise ValueError(f"Cutoff must be between 0 and {nyq:.0f} Hz")
        wn = low / nyq
        btype = "low" if kind == "Low-pass" else "high"
    else:
        if not 0 < low < high < nyq:
            raise ValueError(f"Need 0 < low < high < {nyq:.0f} Hz")
        wn = [low / nyq, high / nyq]
        btype = "bandpass" if kind == "Band-pass" else "bandstop"
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfiltfilt(sos, x)


def compute_fft(x, fs):
    """Single-sided amplitude spectrum -> (freqs, magnitude)."""
    n = len(x)
    window = np.hanning(n)
    spec = np.fft.rfft(x * window)
    mag = 2.0 * np.abs(spec) / window.sum()
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return freqs, mag


def compute_spectrogram(x, fs):
    nperseg = min(256, len(x))
    f, t, sxx = signal.spectrogram(x, fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f, t, 10 * np.log10(sxx + 1e-12)


def snr_db(clean, noisy):
    noise = noisy - clean
    p_noise = np.mean(noise ** 2)
    if p_noise == 0:
        return float("inf")
    return 10 * np.log10(np.mean(clean ** 2) / p_noise)
