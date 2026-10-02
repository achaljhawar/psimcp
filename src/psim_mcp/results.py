"""In-memory store of simulation waveforms plus waveform analysis helpers."""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Waveforms:
    id: str
    source: str  # schematic or graph file the data came from
    names: list[str]  # names[0] is the x-axis (Time for transient, frequency for AC sweeps)
    data: list[np.ndarray]
    graph_file: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> np.ndarray:
        return self.data[0]

    def curve(self, name: str) -> np.ndarray:
        try:
            return self.data[self.names.index(name)]
        except ValueError:
            lower = [n.lower() for n in self.names]
            if name.lower() in lower:
                return self.data[lower.index(name.lower())]
            raise KeyError(f"Curve '{name}' not found. Available: {self.names}") from None

    def window(self, x_start: float | None, x_end: float | None) -> np.ndarray:
        mask = np.ones(len(self.x), dtype=bool)
        if x_start is not None:
            mask &= self.x >= x_start
        if x_end is not None:
            mask &= self.x <= x_end
        if not mask.any():
            raise ValueError(f"No samples in window [{x_start}, {x_end}]; x range is [{self.x[0]}, {self.x[-1]}].")
        return mask

    def steady_state_start(self, fraction: float) -> float:
        return float(self.x[-1] - (self.x[-1] - self.x[0]) * fraction)


def graph_to_waveforms(result_id: str, graph: Any, source: str, graph_file: str | None, parameters: dict) -> Waveforms:
    """Copy a PSIM_Graph out of DLL-owned memory into numpy arrays."""
    names, data = [], []
    for c in graph.curves:
        names.append(str(c.Name))
        data.append(np.array(c.Values[: c.Rows], dtype=float))
    return Waveforms(result_id, source, names, data, graph_file, dict(parameters))


def stats(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Mean and RMS are time-weighted (trapezoidal), since PSIM may output non-uniform steps."""
    span = float(x[-1] - x[0]) if len(x) > 1 else 0.0
    if span > 0:
        mean = float(np.trapezoid(y, x) / span)
        rms = float(np.sqrt(np.trapezoid(y * y, x) / span))
    else:
        mean = float(np.mean(y))
        rms = float(np.sqrt(np.mean(y * y)))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    return {"mean": mean, "rms": rms, "min": ymin, "max": ymax, "peak_to_peak": ymax - ymin}


def fundamental_frequency(y: np.ndarray, x: np.ndarray) -> float | None:
    """Estimate frequency from rising crossings of the mean level."""
    centered = y - np.mean(y)
    rising = np.where((centered[:-1] < 0) & (centered[1:] >= 0))[0]
    if len(rising) < 2:
        return None
    # interpolate crossing instants for accuracy
    t = x[rising] - centered[rising] * (x[rising + 1] - x[rising]) / (centered[rising + 1] - centered[rising])
    return float((len(t) - 1) / (t[-1] - t[0]))


def segment(x: np.ndarray, y: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Samples of y on [a, b], with values interpolated exactly at a and b (like SIMVIEW cursors)."""
    inside = (x > a) & (x < b)
    xs = np.concatenate(([a], x[inside], [b]))
    ys = np.concatenate(([np.interp(a, x, y)], y[inside], [np.interp(b, x, y)]))
    return xs, ys


def spectrum(y: np.ndarray, x: np.ndarray, a: float, b: float) -> tuple[float, np.ndarray, np.ndarray]:
    """One-sided spectrum of y over [a, b]: (bin width in Hz, peak amplitudes, phases in degrees).

    Bin k is at k * bin width; bin 0 holds the DC value. Phases are cosine-referenced to t = a.
    Data are resampled uniformly first, since PSIM may output non-uniform steps.
    """
    xs, ys = segment(x, y, a, b)
    n = max(len(xs) - 1, 8)
    t = a + np.arange(n) * (b - a) / n
    c = np.fft.rfft(np.interp(t, xs, ys)) / n
    amp = 2 * np.abs(c)
    amp[0] = c[0].real
    return 1.0 / (b - a), amp, np.degrees(np.angle(c))


def dominant_frequency(y: np.ndarray, x: np.ndarray) -> float | None:
    """Frequency of the largest non-DC component. Unlike mean-crossing counting, switching ripple
    does not fool it. Hann window plus zero padding gives sub-bin accuracy."""
    n = len(x)
    if n < 16 or x[-1] <= x[0]:
        return None
    t = np.linspace(x[0], x[-1], n)
    yu = np.interp(t, x, y)
    nfft = 1 << int(np.ceil(np.log2(n * 16)))
    mag = np.abs(np.fft.rfft((yu - yu.mean()) * np.hanning(n), nfft))
    k_min = 2 * nfft // n  # skip the Hann main lobe around DC (slow drift, residual offset)
    if k_min >= len(mag) - 1 or not mag[k_min:].any():
        return None
    k = k_min + int(np.argmax(mag[k_min:]))
    coarse = k * (n - 1) / (x[-1] - x[0]) / nfft
    refined = _refine_period(y, x, coarse) or float(coarse)
    # Mean crossings are exact for clean waveforms and unaffected by amplitude changes, but ripple adds
    # spurious crossings; trust them only when they agree with the ripple-proof estimate.
    crossing = fundamental_frequency(y, x)
    return crossing if crossing and abs(crossing - refined) < 0.01 * refined else refined


def _refine_period(y: np.ndarray, x: np.ndarray, f0: float) -> float | None:
    """Refine f0 to the frequency whose period the waveform best repeats with, i.e. the lag near 1/f0
    minimizing mean (y(t + lag) - y(t))^2. Harmonics and synchronous switching ripple do not bias this."""
    t = np.linspace(x[0], x[-1] - 1.25 / f0, 2000)
    if t[-1] <= t[0]:
        return None
    y0 = np.interp(t, x, y)

    def best(lags: np.ndarray) -> int:
        return int(np.argmin([np.mean((np.interp(t + lag, x, y) - y0) ** 2) for lag in lags]))

    lags = np.linspace(0.8 / f0, 1.25 / f0, 451)
    for _ in range(3):  # each pass zooms in 100x around the best lag
        i = best(lags)
        lags = np.linspace(lags[max(i - 1, 0)], lags[min(i + 1, len(lags) - 1)], 201)
    return float(1 / lags[best(lags)])


def harmonic_analysis(
    y: np.ndarray, x: np.ndarray, a: float, b: float, fundamental_hz: float | None, max_harmonic: int
) -> dict[str, Any]:
    """DC, harmonic amplitudes/phases and THD over [a, b]. Without fundamental_hz, the largest non-DC bin is used."""
    df, amp, phase = spectrum(y, x, a, b)
    if fundamental_hz is None:
        fundamental_hz = (1 + int(np.argmax(amp[1:]))) * df
    rows = []
    for h in range(1, max_harmonic + 1):
        k = int(round(h * fundamental_hz / df))
        if k >= len(amp):
            break
        rows.append((h, k * df, float(amp[k]), float(phase[k])))
    if not rows or rows[0][1] == 0:
        raise ValueError(f"Window [{a}, {b}] is too short to resolve {fundamental_hz} Hz; widen it.")
    a1 = rows[0][2]
    thd = float(np.sqrt(sum(r[2] ** 2 for r in rows[1:])) / a1 * 100) if a1 > 0 else None
    return {
        "fundamental_hz": fundamental_hz,
        "cycles_in_window": (b - a) * fundamental_hz,
        "dc": float(amp[0]),
        "harmonics": rows,  # (h, hz, peak amplitude, phase deg)
        "thd_percent": thd,
    }


def summarize(wf: Waveforms, steady_state_fraction: float) -> dict[str, Any]:
    ss_mask = wf.window(wf.steady_state_start(steady_state_fraction), None)
    curves = {}
    for name, y in zip(wf.names[1:], wf.data[1:]):
        ss = stats(y[ss_mask], wf.x[ss_mask])
        curves[name] = {
            "final": _r(y[-1]),
            "ss_mean": _r(ss["mean"]),
            "ss_rms": _r(ss["rms"]),
            "ss_ripple_pp": _r(ss["peak_to_peak"]),
            "overall_min": _r(np.min(y)),
            "overall_max": _r(np.max(y)),
        }
    return {
        "result_id": wf.id,
        "x_axis": wf.names[0],
        "x_range": [_r(wf.x[0]), _r(wf.x[-1])],
        "points": len(wf.x),
        "steady_state_window": [_r(wf.x[ss_mask][0]), _r(wf.x[-1])],
        "curves": curves,
        "graph_file": wf.graph_file,
    }


def _r(v: float) -> float:
    """Round to 6 significant digits to keep responses compact."""
    v = float(v)
    return float(f"{v:.6g}") if np.isfinite(v) else v


class ResultStore:
    def __init__(self, capacity: int = 200):
        self._items: OrderedDict[str, Waveforms] = OrderedDict()
        self._capacity = capacity
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return f"r{next(self._counter)}"

    def put(self, wf: Waveforms) -> None:
        with self._lock:
            self._items[wf.id] = wf
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def get(self, result_id: str) -> Waveforms:
        with self._lock:
            if result_id not in self._items:
                raise KeyError(
                    f"Unknown result_id '{result_id}' (it may have been evicted). "
                    "Re-run the simulation or load its graph file with load_graph_file."
                )
            return self._items[result_id]

    def list(self) -> list[Waveforms]:
        with self._lock:
            return list(self._items.values())
