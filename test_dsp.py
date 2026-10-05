import numpy as np
import pytest
import dsp


def test_lowpass_removes_high_tone():
    fs = 2000
    t = np.arange(0, 1, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 200 * t)
    y = dsp.apply_filter(x, fs, "Low-pass", 50, 0, 4)
    f, m = dsp.compute_fft(y, fs)
    assert m[np.argmin(abs(f - 10))] > 0.8
    assert m[np.argmin(abs(f - 200))] < 0.01


def test_filter_improves_snr():
    t, clean = dsp.generate("Sine", 10, 1, 2000, 1.0)
    _, noisy = dsp.generate("Sine", 10, 1, 2000, 1.0, noise_level=0.5, seed=1)
    y = dsp.apply_filter(noisy, 2000, "Low-pass", 30)
    assert dsp.snr_db(clean, y) > dsp.snr_db(clean, noisy)


def test_invalid_cutoff_raises():
    with pytest.raises(ValueError):
        dsp.apply_filter(np.zeros(100), 1000, "Low-pass", 600)
    with pytest.raises(ValueError):
        dsp.apply_filter(np.zeros(100), 1000, "Band-pass", 100, 50)
