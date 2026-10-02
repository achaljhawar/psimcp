"""Checks of the FFT / cursor math against signals with known harmonics. No PSIM needed.

Run with:  uv run python tests/test_analysis.py
"""

import numpy as np

from psim_mcp.results import dominant_frequency, harmonic_analysis, segment, stats

F1 = 50.0


def signal(t: np.ndarray) -> np.ndarray:
    """1 V DC + 10 V fundamental at 30 deg + 2 V 5th harmonic + 1 V 7th harmonic at -90 deg."""
    w = 2 * np.pi * F1
    return 1 + 10 * np.cos(w * t + np.radians(30)) + 2 * np.cos(5 * w * t) + np.cos(7 * w * t - np.pi / 2)


def nonuniform_time(t_end: float, n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sort(np.concatenate(([0.0, t_end], rng.uniform(0, t_end, n))))


def test_harmonics_on_whole_cycles() -> None:
    x = nonuniform_time(0.1, 40000)
    h = harmonic_analysis(signal(x), x, 0.04, 0.08, F1, 10)  # two whole cycles
    amps = {k: amp for k, _, amp, _ in h["harmonics"]}
    phases = {k: ph for k, _, _, ph in h["harmonics"]}
    assert abs(h["dc"] - 1) < 1e-3
    assert abs(amps[1] - 10) < 1e-3 and abs(amps[5] - 2) < 1e-3 and abs(amps[7] - 1) < 1e-3
    assert amps[3] < 1e-3
    # Phase is referenced to the window start; 0.04 s is two whole cycles, so it matches t = 0.
    assert abs(phases[1] - 30) < 0.05 and abs(phases[7] + 90) < 0.05
    assert abs(h["thd_percent"] - np.sqrt(2**2 + 1**2) / 10 * 100) < 1e-2
    assert abs(h["cycles_in_window"] - 2) < 1e-9


def test_dominant_bin_when_fundamental_unknown() -> None:
    x = np.linspace(0, 0.1, 20001)
    h = harmonic_analysis(signal(x), x, 0.0, 0.1, None, 1)
    assert abs(h["fundamental_hz"] - F1) < 1e-9


def test_estimate_ignores_switching_ripple() -> None:
    # 50 Hz inverter current with 10 kHz triangular ripple: mean-crossing counting sees extra "cycles" here.
    x = nonuniform_time(0.12, 200000)
    ripple = 0.8 * (2 * np.abs((x * 10e3) % 1 - 0.5) - 0.5)
    y = 4 * np.sin(2 * np.pi * F1 * x) + ripple
    f = dominant_frequency(y, x)
    assert f is not None and abs(f - F1) < 1e-3, f


def test_cursor_segment_and_stats() -> None:
    x = np.linspace(0, 1, 11)
    y = 2 * x
    xs, ys = segment(x, y, 0.25, 0.75)
    assert xs[0] == 0.25 and xs[-1] == 0.75
    assert abs(ys[0] - 0.5) < 1e-12 and abs(ys[-1] - 1.5) < 1e-12
    assert abs(stats(ys, xs)["mean"] - 1.0) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
