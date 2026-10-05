# Signal Lab

An interactive desktop app for exploring signal processing, built with Python, Tkinter, NumPy, SciPy and Matplotlib.

## Features
- Generate sine, square, sawtooth, triangle and two-tone waveforms
- Add white noise and see the effect on SNR
- Butterworth filters: low-pass, high-pass, band-pass, band-stop (zero-phase)
- Live plots: time domain, FFT spectrum, spectrogram
- Load a `.wav` file and filter it
- Export the plots as PNG

## Setup
```bash
git clone <your-repo-url>
cd signal-lab
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
On Linux you may also need Tk: `sudo apt install python3-tk`

## Project structure
| File | Purpose |
|------|---------|
| `app.py` | Tkinter UI and plotting |
| `dsp.py` | Signal generation, filtering, FFT (no UI code) |
| `test_dsp.py` | Unit tests (`pytest`) |

## Ideas to extend
- Real-time microphone input (`sounddevice`)
- FIR filter design and frequency-response plot
- Audio playback of original vs filtered signal
- Add a signal-denoising comparison (moving average vs Butterworth)

## License
MIT
