"""Signal Lab - interactive signal generator, filter and spectrum analyzer."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.io import wavfile
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import dsp


class SignalLab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signal Lab")
        self.geometry("1200x780")
        self.minsize(980, 640)

        self.fs = 2000
        self.loaded = None  # (t, x, fs) when a WAV is loaded
        self.vars = {
            "wave": tk.StringVar(value="Two-tone (f + 2.5f)"),
            "freq": tk.DoubleVar(value=10),
            "amp": tk.DoubleVar(value=1.0),
            "noise": tk.DoubleVar(value=0.4),
            "filter": tk.StringVar(value="Low-pass"),
            "low": tk.DoubleVar(value=20),
            "high": tk.DoubleVar(value=60),
            "order": tk.IntVar(value=4),
        }
        self._build_ui()
        self.update_plots()

    # ---------- UI ----------
    def _build_ui(self):
        panel = ttk.Frame(self, padding=10)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        src = ttk.LabelFrame(panel, text="Source", padding=8)
        src.pack(fill=tk.X, pady=4)
        ttk.Label(src, text="Waveform").pack(anchor="w")
        cb = ttk.Combobox(src, textvariable=self.vars["wave"], values=dsp.WAVEFORMS,
                          state="readonly")
        cb.pack(fill=tk.X)
        cb.bind("<<ComboboxSelected>>", lambda e: self._use_generated())
        self._slider(src, "Frequency (Hz)", "freq", 1, 200)
        self._slider(src, "Amplitude", "amp", 0.1, 3)
        self._slider(src, "Noise level", "noise", 0, 2)
        ttk.Button(src, text="Load WAV file...", command=self.load_wav).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(src, text="Back to generator", command=self._use_generated).pack(fill=tk.X, pady=2)

        flt = ttk.LabelFrame(panel, text="Filter (Butterworth)", padding=8)
        flt.pack(fill=tk.X, pady=4)
        ttk.Label(flt, text="Type").pack(anchor="w")
        cb2 = ttk.Combobox(flt, textvariable=self.vars["filter"], values=dsp.FILTERS,
                           state="readonly")
        cb2.pack(fill=tk.X)
        cb2.bind("<<ComboboxSelected>>", lambda e: self.update_plots())
        self._slider(flt, "Cutoff / Low (Hz)", "low", 1, 900)
        self._slider(flt, "High cutoff (Hz)", "high", 2, 950)
        self._slider(flt, "Order", "order", 1, 10, integer=True)

        self.info = ttk.Label(panel, text="", justify=tk.LEFT, wraplength=250)
        self.info.pack(fill=tk.X, pady=8)

        ttk.Button(panel, text="Save plot as PNG", command=self.save_png).pack(fill=tk.X, pady=2)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8, 7), dpi=100, constrained_layout=True)
        self.ax_time = self.fig.add_subplot(3, 1, 1)
        self.ax_fft = self.fig.add_subplot(3, 1, 2)
        self.ax_spec = self.fig.add_subplot(3, 1, 3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame)

    def _slider(self, parent, label, key, lo, hi, integer=False):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        val = ttk.Label(row, width=6, anchor="e")
        val.pack(side=tk.RIGHT)

        def on_change(_=None):
            v = self.vars[key].get()
            if integer:
                v = int(round(v))
                self.vars[key].set(v)
            val.config(text=f"{v:.1f}" if not integer else str(v))
            self.update_plots()

        s = ttk.Scale(parent, from_=lo, to=hi, variable=self.vars[key], command=on_change)
        s.pack(fill=tk.X)
        val.config(text=f"{self.vars[key].get():.1f}" if not integer else str(self.vars[key].get()))

    # ---------- actions ----------
    def load_wav(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        try:
            fs, data = wavfile.read(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not read file:\n{exc}")
            return
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(float)
        peak = np.max(np.abs(data)) or 1.0
        data = data / peak
        data = data[: fs * 5]  # first 5 seconds keeps the UI snappy
        t = np.arange(len(data)) / fs
        self.loaded = (t, data, fs)
        self.update_plots()

    def _use_generated(self):
        self.loaded = None
        self.update_plots()

    def save_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG image", "*.png")])
        if path:
            self.fig.savefig(path, dpi=150)

    # ---------- plotting ----------
    def update_plots(self):
        v = self.vars
        if self.loaded:
            t, x, fs = self.loaded
            clean = None
        else:
            fs = self.fs
            t, clean = dsp.generate(v["wave"].get(), v["freq"].get(), v["amp"].get(),
                                    fs, 1.0, 0.0)
            _, x = dsp.generate(v["wave"].get(), v["freq"].get(), v["amp"].get(),
                                fs, 1.0, v["noise"].get(), seed=42)

        try:
            y = dsp.apply_filter(x, fs, v["filter"].get(), v["low"].get(),
                                 v["high"].get(), int(v["order"].get()))
            err = ""
        except ValueError as exc:
            y, err = x, f"Filter skipped: {exc}"

        shown = min(len(t), int(fs * 0.5)) if not self.loaded else min(len(t), int(fs * 0.1))
        self.ax_time.clear()
        self.ax_time.plot(t[:shown], x[:shown], color="#9aa0a6", lw=0.9, label="Input")
        self.ax_time.plot(t[:shown], y[:shown], color="#d93025", lw=1.4, label="Filtered")
        self.ax_time.set_title("Time domain")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.legend(loc="upper right")
        self.ax_time.grid(alpha=0.3)

        f, mx = dsp.compute_fft(x, fs)
        _, my = dsp.compute_fft(y, fs)
        self.ax_fft.clear()
        self.ax_fft.plot(f, mx, color="#9aa0a6", lw=0.9, label="Input")
        self.ax_fft.plot(f, my, color="#d93025", lw=1.4, label="Filtered")
        self.ax_fft.set_xlim(0, min(fs / 2, 300 if not self.loaded else fs / 2))
        self.ax_fft.set_title("Frequency spectrum (FFT)")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Magnitude")
        self.ax_fft.legend(loc="upper right")
        self.ax_fft.grid(alpha=0.3)

        self.ax_spec.clear()
        sf, st, sxx = dsp.compute_spectrogram(y, fs)
        self.ax_spec.pcolormesh(st, sf, sxx, shading="auto", cmap="magma")
        self.ax_spec.set_ylim(0, min(fs / 2, 300 if not self.loaded else fs / 2))
        self.ax_spec.set_title("Spectrogram (filtered)")
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")

        self.canvas.draw_idle()

        lines = [f"Sample rate: {fs} Hz   Samples: {len(x)}"]
        if clean is not None:
            lines.append(f"SNR input:    {dsp.snr_db(clean, x):6.1f} dB")
            lines.append(f"SNR filtered: {dsp.snr_db(clean, y):6.1f} dB")
        if err:
            lines.append(err)
        self.info.config(text="\n".join(lines))


if __name__ == "__main__":
    SignalLab().mainloop()
