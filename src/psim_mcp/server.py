"""MCP server exposing Altair PSIM 2026 simulation, schematic editing and waveform analysis."""

from __future__ import annotations

import contextlib
import csv
import ctypes
import io
import itertools
import os
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .engine import PsimEngine, PsimError
from .results import (
    ResultStore,
    Waveforms,
    _r,
    dominant_frequency,
    fundamental_frequency,
    graph_to_waveforms,
    harmonic_analysis,
    segment,
    stats,
    summarize,
)

INSTRUCTIONS = """\
Drive Altair PSIM 2026 (power-electronics circuit simulator) on this machine.

Typical workflow:
1. inspect_schematic on a .psimsch file to see its elements, parameters, output probes and
   undefined variables (undefined variables MUST be supplied in `parameters` when simulating).
2. simulate it (optionally overriding variables or element values) -> returns a result_id and a
   steady-state summary of every output curve.
3. analyze_waveform / get_waveform on the result_id for details; parameter_sweep for design studies.
   measure_cursors and fft_analysis reproduce SIMVIEW's Measure cursors and FFT as numbers
   (SIMVIEW's own cursors cannot be driven programmatically).
PSIM values accept engineering suffixes as strings (e.g. "50u", "20k", "1.2m").
The first PSIM call loads the PSIM engine and license, which can take ~30-60 s.
"""

OUTPUT_DIR = Path(os.environ.get("PSIM_MCP_OUTPUT_DIR", Path.home() / "psim_mcp_output"))

mcp = MCPServer("psim", instructions=INSTRUCTIONS)
engine = PsimEngine()
store = ResultStore()

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
RUNS_SIM = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
WRITES_FILES = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

Value = str | float | int


class ElementEdit(BaseModel):
    element_type: Annotated[str, Field(description='Element type as shown by inspect_schematic, e.g. "C", "R", "MOSFET".')]
    element_name: Annotated[str, Field(description='Element name, e.g. "C1".')]
    parameter: Annotated[str, Field(description='Parameter name as shown by inspect_schematic, e.g. "Capacitance".')]
    value: Annotated[Value, Field(description='New value; numbers, suffixed strings ("1.2m") or variable names.')]


class ElementToggle(BaseModel):
    element_type: str
    element_name: str
    enabled: bool


class NewElement(BaseModel):
    type: Annotated[str, Field(description='PSIM type keyword, e.g. "SIMCONTROL", "VDC", "R", "L", "C", "MOSFET", "WIRE", ".FILE", "PORT", "SUBCIRCUIT", "TEXT".')]
    name: Annotated[str, Field(description='Element name (empty string for WIRE / SIMCONTROL).')] = ""
    params: Annotated[
        dict[str, Value],
        Field(
            description='Placement and parameter keywords, e.g. {"PORTS": "{120, 100}", "DIRECTION": 90, "Resistance": 15} '
            'or for wires {"X1": 120, "Y1": 100, "X2": 150, "Y2": 100}.'
        ),
    ] = {}


# ---------------------------------------------------------------- helpers


def _existing(path: str, suffixes: tuple[str, ...] = ()) -> str:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ToolError(f"File not found: {p}")
    if suffixes and p.suffix.lower() not in suffixes:
        raise ToolError(f"Expected a {'/'.join(suffixes)} file, got: {p.name}")
    return str(p)


def _output_path(path: str | None, default_name: str) -> str:
    p = Path(path).expanduser().resolve() if path else OUTPUT_DIR / default_name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _call(fn, *args, **kwargs):
    try:
        return engine.call(fn, *args, **kwargs)
    except PsimError as e:
        raise ToolError(str(e)) from e


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _apply_edits(psim, sch, edits: list[ElementEdit], toggles: list[ElementToggle]) -> None:
    for e in edits:
        if not psim.PsimSetElmValue2(sch, e.element_type, e.element_name, e.parameter, str(e.value)):
            raise PsimError(f"Could not set {e.element_type} '{e.element_name}' {e.parameter} = {e.value}")
    for t in toggles:
        if not psim.PsimEnableElm2(sch, t.element_type, t.element_name, 1 if t.enabled else 0):
            raise PsimError(f"Could not {'enable' if t.enabled else 'disable'} {t.element_type} '{t.element_name}'")


def _sim_error(res, printed: str) -> ToolError:
    msg = (res.ErrorMessage if res is not None else "") or printed or "Simulation failed (no message from PSIM)."
    return ToolError(msg.strip())


# ---------------------------------------------------------------- general


@mcp.tool(annotations=READ_ONLY)
def psim_status() -> dict[str, Any]:
    """Load PSIM if needed and report version, install folder and output folder."""
    _call(lambda psim: None)
    return {
        "psim_version": engine.version_name,
        "psim_folder": engine.folder,
        "output_dir": str(OUTPUT_DIR),
        "cached_results": len(store.list()),
    }


@mcp.tool(annotations=READ_ONLY)
def find_examples(
    query: Annotated[str, Field(description='Case-insensitive words to match in the path, e.g. "buck", "llc resonant", "pmsm".')] = "",
    limit: int = 30,
) -> list[str]:
    """Search the PSIM install (examples, tutorials, case studies) for .psimsch schematics.

    Examples are useful templates: copy one before editing it in place.
    """
    words = query.lower().split()
    hits = []
    for p in Path(engine.folder).rglob("*.psimsch"):
        s = str(p)
        if all(w in s.lower() for w in words):
            hits.append(s)
            if len(hits) >= limit:
                break
    return hits


# ---------------------------------------------------------------- schematic inspection


@mcp.tool(annotations=READ_ONLY)
def inspect_schematic(
    schematic_path: str,
    element_types: Annotated[
        list[str] | None, Field(description='Only list elements of these types, e.g. ["R", "C", "L"]. Default: all.')
    ] = None,
    include_parameters: bool = True,
) -> dict[str, Any]:
    """List a schematic's elements and their parameters, its output probes, and undefined variables.

    Undefined variables must be passed via `parameters` when calling simulate.
    Element parameter names shown here are what simulate/edit_schematic `element_edits` expect.
    """
    path = _existing(schematic_path, (".psimsch", ".sch"))
    wanted = {t.lower() for t in element_types} if element_types else None

    def run(psim):
        sch = engine.open_schematic(path)
        try:
            elements = psim.PsimGetElementList(sch, 0) or []
            _, probes, _, _ = psim.Psim_GetProbes(sch)
            _, undefined, _, _ = psim.Psim_GetUndefinedParameters(sch)
            is_sub = psim.PsimIsSubcircuit(sch)
        finally:
            sch.free()
        out = []
        for e in elements:
            if wanted and e.Type.lower() not in wanted:
                continue
            item: dict[str, Any] = {"type": e.Type, "name": e.Name}
            if include_parameters:
                item["parameters"] = {p.Name: p.Value for p in e.Params if p.Name != "Name"}
            out.append(item)
        return {
            "schematic": path,
            "is_subcircuit": bool(is_sub),
            "output_probes": _split_lines(probes),
            "undefined_variables": _split_lines(undefined),
            "element_count": len(elements),
            "elements": out,
        }

    return _call(run)[0]


@mcp.tool(annotations=READ_ONLY)
def schematic_to_python(schematic_path: str) -> str:
    """Convert a schematic into the equivalent PSIM Python build script (PsimCreateNewElement calls).

    This reveals full topology: element placement, port coordinates and wires. Use it to understand
    how a circuit is connected, or as a template for create_schematic.
    """
    path = _existing(schematic_path, (".psimsch", ".sch"))
    with tempfile.TemporaryDirectory() as tmp:
        py = os.path.join(tmp, "schematic.py")
        ok, printed = _call(lambda psim: psim.PsimConvertToPython(path, py))
        if not ok or not os.path.isfile(py):
            raise ToolError(printed or f"PSIM could not convert {path}")
        return Path(py).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- simulation


@mcp.tool(annotations=RUNS_SIM)
def simulate(
    schematic_path: str,
    parameters: Annotated[
        dict[str, Value] | None,
        Field(
            description='Variable overrides, e.g. {"Vin": 48, "Fsw": "100k"}. Also accepts TotalTime and TimeStep '
            "to override the Simulation Control."
        ),
    ] = None,
    element_edits: Annotated[
        list[ElementEdit] | None, Field(description="Element parameter changes applied in memory only; the file is not modified.")
    ] = None,
    element_toggles: Annotated[list[ElementToggle] | None, Field(description="Enable/disable elements for this run only.")] = None,
    engine_name: Annotated[Literal["psim", "hyperspice", "ltspice"], Field(description="Solver to use.")] = "psim",
    graph_file: Annotated[str | None, Field(description="Where to write the .smv/.txt/.csv results. Default: output dir.")] = None,
    steady_state_fraction: Annotated[
        float, Field(gt=0, le=1, description="Trailing fraction of the run used for the ss_* statistics.")
    ] = 0.2,
    open_simview: Annotated[bool, Field(description="Also open the results in the SIMVIEW GUI.")] = False,
) -> dict[str, Any]:
    """Run a transient simulation and return a result_id plus a summary of each output curve
    (final value, steady-state mean/RMS/peak-to-peak ripple, overall min/max)."""
    path = _existing(schematic_path, (".psimsch", ".sch"))
    params = dict(parameters or {})
    rid = store.new_id()
    out = _output_path(graph_file, f"{Path(path).stem}_{rid}.smv")
    kwargs = {k: str(v) for k, v in params.items()}
    kwargs["Simview"] = 1 if open_simview else 0

    def run(psim):
        fn = {
            "psim": psim.PsimSimulate,
            "hyperspice": psim.PsimSimulate_HyperSpice,
            "ltspice": psim.PsimSimulate_LTSpice,
        }[engine_name]
        target: Any = path
        if element_edits or element_toggles:
            target = engine.open_schematic(path)
            _apply_edits(psim, target, element_edits or [], element_toggles or [])
        try:
            res = fn(target, out, **kwargs)
            if res is None or res.Result != 1 or len(res.Graph) == 0:
                return None, res
            return graph_to_waveforms(rid, res.Graph, path, out, params), res
        finally:
            if not isinstance(target, str):
                target.free()

    (wf, res), printed = _call(run)
    if wf is None:
        raise _sim_error(res, printed)
    store.put(wf)
    summary = summarize(wf, steady_state_fraction)
    if res.ErrorMessage:
        summary["warnings"] = res.ErrorMessage.strip()
    return summary


@mcp.tool(annotations=RUNS_SIM)
def parameter_sweep(
    schematic_path: str,
    sweep: Annotated[
        dict[str, list[Value]],
        Field(description='Variables to sweep, e.g. {"L": ["20u", "50u"], "Fsw": ["100k", "200k"]}.'),
    ],
    mode: Annotated[
        Literal["grid", "zip"],
        Field(description='"grid" = every combination; "zip" = pair the i-th values of each list (equal lengths).'),
    ] = "grid",
    base_parameters: Annotated[dict[str, Value] | None, Field(description="Fixed variable overrides for every run.")] = None,
    curves: Annotated[list[str] | None, Field(description="Curves to report per run. Default: all.")] = None,
    max_threads: Annotated[int, Field(ge=1, le=64, description="Concurrent simulations.")] = 8,
    steady_state_fraction: Annotated[float, Field(gt=0, le=1)] = 0.2,
) -> dict[str, Any]:
    """Run many simulations in parallel over a set of variable values and tabulate each output curve's
    steady-state mean and peak-to-peak ripple per run. Each run gets its own result_id for drill-down."""
    path = _existing(schematic_path, (".psimsch", ".sch"))
    names = list(sweep)
    if mode == "zip":
        lengths = {len(v) for v in sweep.values()}
        if len(lengths) != 1:
            raise ToolError('"zip" mode needs all sweep lists to have the same length.')
        combos = list(zip(*sweep.values()))
    else:
        combos = list(itertools.product(*sweep.values()))
    if not combos:
        raise ToolError("Sweep is empty.")
    if len(combos) > 1000:
        raise ToolError(f"Sweep has {len(combos)} runs; limit is 1000. Narrow it down.")

    runs = []
    stem = Path(path).stem
    for combo in combos:
        rid = store.new_id()
        params = {**(base_parameters or {}), **dict(zip(names, combo))}
        runs.append((rid, params, _output_path(None, f"{stem}_{rid}.smv")))

    def run(psim):
        data = [psim.SimulateData_from_dict(path, out, {**{k: str(v) for k, v in p.items()}, "Simview": 0}) for _, p, out in runs]
        results = psim.PsimASimulate(max_threads, data)
        if results is None or len(results) != len(runs):
            # PsimASimulate can misalign results when a run fails to initialise; fall back to sequential.
            results = [psim.PsimSimulate(path, out, Simview=0, **{k: str(v) for k, v in p.items()}) for _, p, out in runs]
        converted = []
        for (rid, p, out), res in zip(runs, results):
            if res is not None and res.Result == 1 and len(res.Graph) > 0:
                converted.append((graph_to_waveforms(rid, res.Graph, path, out, p), None))
            else:
                converted.append((None, (res.ErrorMessage if res is not None else "") or "failed"))
        return converted

    converted, _ = _call(run)
    table = []
    for (rid, params, _), (wf, err) in zip(runs, converted):
        row: dict[str, Any] = {"parameters": params}
        if wf is None:
            row["error"] = err.strip()[:500]
        else:
            store.put(wf)
            s = summarize(wf, steady_state_fraction)
            row["result_id"] = rid
            row["curves"] = {
                n: {"ss_mean": c["ss_mean"], "ss_ripple_pp": c["ss_ripple_pp"]}
                for n, c in s["curves"].items()
                if not curves or n in curves
            }
        table.append(row)
    ok = sum("error" not in r for r in table)
    return {"schematic": path, "runs": len(table), "succeeded": ok, "failed": len(table) - ok, "results": table}


# ---------------------------------------------------------------- results


@mcp.tool(annotations=READ_ONLY)
def analyze_waveform(
    result_id: str,
    curve: str,
    x_start: Annotated[float | None, Field(description="Window start (seconds for transient). Default: beginning.")] = None,
    x_end: Annotated[float | None, Field(description="Window end. Default: end of run.")] = None,
) -> dict[str, Any]:
    """Statistics of one curve over a time window: mean, RMS, AC RMS, min, max, peak-to-peak,
    value at window end, and estimated fundamental frequency."""
    wf = _get(result_id)
    y = _curve(wf, curve)
    mask = _window(wf, x_start, x_end)
    x, y = wf.x[mask], y[mask]
    s = stats(y, x)
    freq = fundamental_frequency(y, x)
    return {
        "result_id": result_id,
        "curve": curve,
        "window": [_r(x[0]), _r(x[-1])],
        "points": int(len(x)),
        **{k: _r(v) for k, v in s.items()},
        "ac_rms": _r(np.sqrt(max(s["rms"] ** 2 - s["mean"] ** 2, 0.0))),
        "value_at_end": _r(y[-1]),
        "fundamental_hz": _r(freq) if freq else None,
    }


@mcp.tool(annotations=READ_ONLY)
def get_waveform(
    result_id: str,
    curves: Annotated[list[str] | None, Field(description="Curve names. Default: all.")] = None,
    x_start: float | None = None,
    x_end: float | None = None,
    max_points: Annotated[int, Field(ge=2, le=5000, description="Evenly downsample to at most this many points.")] = 300,
) -> dict[str, Any]:
    """Return (downsampled) waveform samples for plotting or detailed inspection.
    Downsampling can hide switching ripple; narrow the window to see it."""
    wf = _get(result_id)
    names = curves or wf.names[1:]
    ys = {n: _curve(wf, n) for n in names}
    mask = _window(wf, x_start, x_end)
    idx = np.flatnonzero(mask)
    if len(idx) > max_points:
        idx = idx[np.linspace(0, len(idx) - 1, max_points).round().astype(int)]
    return {
        "result_id": result_id,
        "x_axis": wf.names[0],
        "downsampled": bool(len(idx) < mask.sum()),
        "x": [_r(v) for v in wf.x[idx]],
        "curves": {n: [_r(v) for v in y[idx]] for n, y in ys.items()},
    }


@mcp.tool(annotations=READ_ONLY)
def list_results() -> list[dict[str, Any]]:
    """List cached simulation results (most recent last)."""
    return [
        {
            "result_id": wf.id,
            "source": wf.source,
            "parameters": wf.parameters,
            "curves": wf.names[1:],
            "points": len(wf.x),
            "graph_file": wf.graph_file,
        }
        for wf in store.list()
    ]


@mcp.tool(annotations=RUNS_SIM)
def load_graph_file(graph_path: str, steady_state_fraction: Annotated[float, Field(gt=0, le=1)] = 0.2) -> dict[str, Any]:
    """Load an existing PSIM/SIMVIEW result file (.smv, .txt, .csv, ...) so it can be analyzed like a simulation result."""
    path = _existing(graph_path)
    rid = store.new_id()

    def run(psim):
        res = psim.PsimReadGraphFile(path)
        if res.Result != 1 or len(res.Graph) == 0:
            return None, res
        return graph_to_waveforms(rid, res.Graph, path, path, {}), res

    (wf, res), printed = _call(run)
    if wf is None:
        raise _sim_error(res, printed)
    store.put(wf)
    return summarize(wf, steady_state_fraction)


@mcp.tool(annotations=WRITES_FILES)
def export_csv(
    result_id: str,
    output_path: Annotated[str | None, Field(description="Destination .csv. Default: output dir.")] = None,
    curves: list[str] | None = None,
) -> str:
    """Write a result's full-resolution waveforms to CSV and return the file path."""
    wf = _get(result_id)
    names = [wf.names[0], *(curves or wf.names[1:])]
    cols = [wf.x, *[_curve(wf, n) for n in names[1:]]]
    out = _output_path(output_path, f"{result_id}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(zip(*(c.tolist() for c in cols)))
    return out


@mcp.tool(annotations=RUNS_SIM)
def open_in_simview(
    graph_path: str,
    curves: Annotated[list[str] | None, Field(description="Curves to display initially.")] = None,
    fft: Annotated[bool, Field(description="Open straight into SIMVIEW's FFT (harmonic spectrum) view.")] = False,
    x_log: bool = False,
    y_log: bool = False,
    x_start: Annotated[float | None, Field(description="Only show data from here (writes a trimmed copy).")] = None,
    x_end: Annotated[float | None, Field(description="Only show data up to here (writes a trimmed copy).")] = None,
) -> str:
    """Open a result file in PSIM's SIMVIEW waveform viewer on the user's desktop.

    SIMVIEW's FFT covers the whole displayed range, so with fft=True pass x_start/x_end spanning a whole
    number of fundamental cycles in steady state. SIMVIEW's measure cursors cannot be placed
    programmatically; use measure_cursors and fft_analysis to get those numbers.
    """
    path = _existing(graph_path)
    options = _simview_options(fft, x_log, y_log)
    if x_start is None and x_end is None:
        _call(lambda psim: psim.RunSimview(path, *(curves or []), **options))
        return f"Opened {path} in SIMVIEW" + (" (FFT view)." if fft else ".")

    def run(psim):
        res = psim.PsimReadGraphFile(path)
        if res.Result != 1 or len(res.Graph) == 0:
            raise PsimError(res.ErrorMessage or f"Could not read {path}")
        wf = graph_to_waveforms("", res.Graph, path, path, {})
        a, b = _clamp_window(wf, x_start, x_end)
        out = _output_path(None, f"{Path(path).stem}_{a:.6g}-{b:.6g}s.smv")
        _open_window_in_simview(psim, wf, curves or wf.names[1:], a, b, out, options)
        return out, a, b

    (out, a, b), _ = _call(run)
    return f"Opened {out} (data {_r(a)} to {_r(b)} from {path}) in SIMVIEW" + (" (FFT view)." if fft else ".")


# ---------------------------------------------------------------- SIMVIEW-style measurements


@mcp.tool(annotations=READ_ONLY)
def measure_cursors(
    result_id: str,
    x1: Annotated[float, Field(description="First cursor position (seconds for transient results).")],
    x2: Annotated[float, Field(description="Second cursor position.")],
    curves: Annotated[list[str] | None, Field(description="Curves to measure. Default: all.")] = None,
    voltage_curve: Annotated[str | None, Field(description="With current_curve: report real/apparent power and PF.")] = None,
    current_curve: str | None = None,
    fundamental_hz: Annotated[
        float | None, Field(gt=0, description="If set, also report THD over the one fundamental cycle ending at x2.")
    ] = None,
) -> dict[str, Any]:
    """Place two SIMVIEW-style measure cursors and read them out: each curve's value at x1 and x2 and their
    difference, 1/dx, and between the cursors the average, RMS, average of |x|, min and max (with location).
    Optionally real power P, apparent power S and power factor P/S of a voltage/current pair, and THD
    as SIMVIEW computes it."""
    wf = _get(result_id)
    a, b = sorted((x1, x2))
    if a == b:
        raise ToolError("x1 and x2 must differ.")
    if a < wf.x[0] or b > wf.x[-1]:
        raise ToolError(f"Cursors must lie within the x range [{_r(wf.x[0])}, {_r(wf.x[-1])}].")
    names = curves or wf.names[1:]
    out: dict[str, Any] = {"result_id": result_id, "x1": _r(a), "x2": _r(b), "dx": _r(b - a), "inv_dx": _r(1 / (b - a))}
    table = {}
    for name in names:
        xs, ys = segment(wf.x, _curve(wf, name), a, b)
        s = stats(ys, xs)
        row = {
            "at_x1": _r(ys[0]),
            "at_x2": _r(ys[-1]),
            "delta": _r(ys[-1] - ys[0]),
            "avg": _r(s["mean"]),
            "rms": _r(s["rms"]),
            "avg_abs": _r(np.trapezoid(np.abs(ys), xs) / (b - a)),
            "min": _r(s["min"]),
            "x_at_min": _r(xs[np.argmin(ys)]),
            "max": _r(s["max"]),
            "x_at_max": _r(xs[np.argmax(ys)]),
        }
        if fundamental_hz:
            start = b - 1 / fundamental_hz
            if start < wf.x[0]:
                raise ToolError(f"THD needs one full cycle ({_r(1 / fundamental_hz)}) of data before x2.")
            thd = harmonic_analysis(_curve(wf, name), wf.x, start, b, fundamental_hz, 50)["thd_percent"]
            row["thd_percent"] = _r(thd) if thd is not None else None
        table[name] = row
    out["curves"] = table
    if voltage_curve or current_curve:
        if not (voltage_curve and current_curve):
            raise ToolError("Pass both voltage_curve and current_curve for power measurements.")
        xs, v = segment(wf.x, _curve(wf, voltage_curve), a, b)
        _, i = segment(wf.x, _curve(wf, current_curve), a, b)
        p = float(np.trapezoid(v * i, xs) / (b - a))
        s_va = stats(v, xs)["rms"] * stats(i, xs)["rms"]
        out["power"] = {
            "voltage": voltage_curve,
            "current": current_curve,
            "real_power": _r(p),
            "apparent_power": _r(s_va),
            "power_factor": _r(p / s_va) if s_va else None,
        }
    return out


@mcp.tool(annotations=RUNS_SIM)
def fft_analysis(
    result_id: str,
    curves: Annotated[list[str], Field(min_length=1, description="Curves to analyze.")],
    fundamental_hz: Annotated[
        float | None,
        Field(gt=0, description="Fundamental frequency. Strongly recommended for PWM/switched waveforms. Default: estimated."),
    ] = None,
    cycles: Annotated[int | None, Field(ge=1, description="Whole fundamental cycles to analyze, ending at x_end.")] = None,
    x_start: Annotated[float | None, Field(description="Window start. Default: whole cycles back from x_end.")] = None,
    x_end: Annotated[float | None, Field(description="Window end. Default: end of run.")] = None,
    max_harmonic: Annotated[int, Field(ge=1, le=1000, description="Highest harmonic reported and included in THD.")] = 50,
    min_percent: Annotated[
        float, Field(ge=0, description="Only list harmonics at least this % of the fundamental (the fundamental is always listed).")
    ] = 0.5,
    steady_state_fraction: Annotated[
        float, Field(gt=0, le=1, description="Without cycles or x_start, fit whole cycles into this trailing fraction of the run.")
    ] = 0.2,
    open_simview: Annotated[bool, Field(description="Also show the analyzed window in SIMVIEW's FFT view.")] = False,
) -> dict[str, Any]:
    """Harmonic spectrum like SIMVIEW's FFT: DC, fundamental and harmonic amplitudes (peak) and phases,
    and THD, over a window of whole fundamental cycles (so there is no spectral leakage).

    Phases are in degrees, cosine-referenced to the window start. With fundamental_hz given the window is
    snapped to whole cycles; with an explicit x_start and x_end the window is used as-is.
    """
    wf = _get(result_id)
    ys = {n: _curve(wf, n) for n in curves}
    first = ys[curves[0]]
    estimated = False
    if fundamental_hz is None and (x_start is None or x_end is None):
        mask = _window(wf, wf.steady_state_start(max(steady_state_fraction, 0.5)), None)
        fundamental_hz = dominant_frequency(first[mask], wf.x[mask])
        if not fundamental_hz:
            raise ToolError(f"Could not estimate the fundamental of '{curves[0]}'; pass fundamental_hz.")
        estimated = True
    if x_start is not None and x_end is not None:
        a, b = x_start, x_end
    else:
        period = 1 / fundamental_hz
        n = cycles or max(1, int(steady_state_fraction * (wf.x[-1] - wf.x[0]) / period))
        if x_start is not None:
            a, b = x_start, x_start + n * period
        else:
            b = wf.x[-1] if x_end is None else x_end
            a = b - n * period
    if a < wf.x[0] or b > wf.x[-1] or b <= a:
        raise ToolError(f"FFT window [{_r(a)}, {_r(b)}] is outside the data range [{_r(wf.x[0])}, {_r(wf.x[-1])}].")
    if fundamental_hz is None:  # explicit window: take the first curve's dominant component
        fundamental_hz = harmonic_analysis(first, wf.x, a, b, None, 1)["fundamental_hz"]
        estimated = True

    results = {}
    for name, y in ys.items():
        try:
            h = harmonic_analysis(y, wf.x, a, b, fundamental_hz, max_harmonic)
        except ValueError as e:
            raise ToolError(str(e)) from None
        (_, f1, a1, p1), rest = h["harmonics"][0], h["harmonics"][1:]
        results[name] = {
            "dc": _r(h["dc"]),
            "fundamental": {"hz": _r(f1), "amplitude": _r(a1), "rms": _r(a1 / np.sqrt(2)), "phase_deg": _r(p1)},
            "thd_percent": _r(h["thd_percent"]) if h["thd_percent"] is not None else None,
            "harmonics": [
                {"h": k, "hz": _r(f), "amplitude": _r(amp), "percent": _r(amp / a1 * 100), "phase_deg": _r(ph)}
                for k, f, amp, ph in rest
                if a1 > 0 and amp / a1 * 100 >= min_percent
            ],
        }
    cycles_in_window = (b - a) * fundamental_hz
    out: dict[str, Any] = {
        "result_id": result_id,
        "window": [_r(a), _r(b)],
        "fundamental_hz": _r(fundamental_hz),
        "fundamental_estimated": estimated,
        "resolution_hz": _r(1 / (b - a)),
        "thd_up_to_harmonic": max_harmonic,
        "curves": results,
    }
    if abs(cycles_in_window - round(cycles_in_window)) > 0.01:
        out["warning"] = (
            f"Window spans {cycles_in_window:.3f} fundamental cycles, not a whole number; expect spectral leakage."
        )
    if open_simview:
        out_path = _output_path(None, f"{result_id}_fft_{a:.6g}-{b:.6g}s.smv")
        _call(lambda psim: _open_window_in_simview(psim, wf, curves, a, b, out_path, _simview_options(True, False, False)))
        out["simview_file"] = out_path
    return out


# ---------------------------------------------------------------- schematic editing


@mcp.tool(annotations=WRITES_FILES)
def edit_schematic(
    schematic_path: str,
    output_path: Annotated[
        str, Field(description="Where to save the modified schematic. Pass the same path as schematic_path to overwrite.")
    ],
    element_edits: list[ElementEdit] | None = None,
    element_toggles: list[ElementToggle] | None = None,
) -> str:
    """Change element parameters and/or enable/disable elements, then save the schematic."""
    path = _existing(schematic_path, (".psimsch", ".sch"))
    if not element_edits and not element_toggles:
        raise ToolError("Nothing to change: pass element_edits and/or element_toggles.")
    out = _output_path(output_path, "")

    def run(psim):
        sch = engine.open_schematic(path)
        try:
            _apply_edits(psim, sch, element_edits or [], element_toggles or [])
            return psim.PsimFileSave(sch, out)
        finally:
            sch.free()

    ok, printed = _call(run)
    if not ok:
        raise ToolError(printed or f"Could not save {out}")
    return f"Saved {out}"


@mcp.tool(annotations=WRITES_FILES)
def create_schematic(
    output_path: Annotated[str, Field(description="Destination .psimsch path.")],
    elements: Annotated[
        list[NewElement],
        Field(
            description="Elements in creation order. Include a SIMCONTROL (TIMESTEP, TOTALTIME). Elements connect where "
            "port coordinates coincide or WIRE endpoints meet. Run schematic_to_python on a similar example first to "
            "learn exact keywords and coordinates."
        ),
    ],
) -> dict[str, Any]:
    """Build a new schematic from a list of elements and wires, and save it."""
    out = _output_path(output_path, "")

    def run(psim):
        sch = psim.PsimFileNew()
        if not sch:
            raise PsimError("PSIM could not create a new schematic.")
        errors = []
        try:
            for i, e in enumerate(elements):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = psim.PsimCreateNewElement(sch, e.type, e.name, **{k: str(v) for k, v in e.params.items()})
                if rc == -1:
                    errors.append({"index": i, "type": e.type, "name": e.name, "error": buf.getvalue().strip()})
            saved = psim.PsimFileSave(sch, out)
        finally:
            sch.free()
        return saved, errors

    (saved, errors), printed = _call(run)
    if not saved:
        raise ToolError(printed or f"Could not save {out}")
    return {"saved": out, "elements_created": len(elements) - len(errors), "errors": errors}


# ---------------------------------------------------------------- design suites & code generation


@mcp.tool(annotations=READ_ONLY)
def design_suite_get_parameters(schematic_path: str) -> dict[str, Any]:
    """Read the design parameters of a PSIM Design Suite schematic (e.g. motor drive, PFC, LLC designs)."""
    path = _existing(schematic_path, (".psimsch", ".sch"))
    (ok, text, count, err), printed = _call(lambda psim: psim.Psim_DS_GetParameters(path))
    if not ok:
        raise ToolError(err or printed or "Not a Design Suite schematic?")
    return {"schematic": path, "count": count, "parameters": text}


@mcp.tool(annotations=WRITES_FILES)
def design_suite_update_parameters(
    schematic_path: str,
    parameters: Annotated[dict[str, Value], Field(description='e.g. {"Vdc": 400, "Pout": "10k"}')],
) -> str:
    """Update Design Suite parameters. PSIM recalculates dependent values and writes them into the design's files."""
    path = _existing(schematic_path, (".psimsch", ".sch"))
    (ok, err), printed = _call(lambda psim: psim.Psim_DS_UpdateParameters(path, dict(parameters)))
    if not ok:
        raise ToolError(err or printed or "Update failed.")
    return f"Updated {len(parameters)} Design Suite parameter(s) in {path}"


@mcp.tool(annotations=WRITES_FILES)
def simcoder_generate_code(
    schematic_path: str,
    output_folder: Annotated[str | None, Field(description="Folder for generated C code. Default: PSIM's default location.")] = None,
) -> str:
    """Generate embedded C code with SimCoder for a schematic set up for a hardware target."""
    path = _existing(schematic_path, (".psimsch", ".sch"))
    out = str(Path(output_folder).expanduser().resolve()) if output_folder else ""
    res, printed = _call(lambda psim: psim.SimcoderGenerateCode(path, out))
    if res is None or res.Result != 1:
        raise _sim_error(res, printed)
    return f"SimCoder code generated for {path}" + (f" into {out}" if out else "")


# ---------------------------------------------------------------- internals


def _get(result_id: str) -> Waveforms:
    try:
        return store.get(result_id)
    except KeyError as e:
        raise ToolError(str(e.args[0])) from None


def _curve(wf: Waveforms, name: str) -> np.ndarray:
    try:
        return wf.curve(name)
    except KeyError as e:
        raise ToolError(str(e.args[0])) from None


def _window(wf: Waveforms, x_start: float | None, x_end: float | None) -> np.ndarray:
    try:
        return wf.window(x_start, x_end)
    except ValueError as e:
        raise ToolError(str(e)) from None


def _clamp_window(wf: Waveforms, x_start: float | None, x_end: float | None) -> tuple[float, float]:
    a = wf.x[0] if x_start is None else max(x_start, wf.x[0])
    b = wf.x[-1] if x_end is None else min(x_end, wf.x[-1])
    if b <= a:
        raise PsimError(f"Empty window [{x_start}, {x_end}]; x range is [{wf.x[0]}, {wf.x[-1]}].")
    return float(a), float(b)


def _simview_options(fft: bool, x_log: bool, y_log: bool) -> dict[str, str]:
    """Name/value options RunSimview passes to SIMVIEW."""
    return {k: "1" for k, on in (("FFT", fft), ("XLOG", x_log), ("YLOG", y_log)) if on}


def _open_window_in_simview(psim, wf: Waveforms, curves: list[str], a: float, b: float, out: str, options: dict) -> None:
    """Write curves over exactly [a, b] to a graph file and open it in SIMVIEW (runs on the PSIM thread).

    SIMVIEW's FFT uses everything on screen, so trimming the file is what pins its FFT to the window.
    """
    try:
        cols = [segment(wf.x, wf.x, a, b)[0], *(segment(wf.x, wf.curve(n), a, b)[1] for n in curves)]
    except KeyError as e:
        raise PsimError(str(e.args[0])) from None
    cols = [np.ascontiguousarray(c, dtype=float) for c in cols]
    ptrs = [c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for c in cols]
    if not psim.PsimWriteGraphFile(out, len(cols), len(cols[0]), [wf.names[0], *curves], ptrs):
        raise PsimError(f"Could not write {out}")
    psim.RunSimview(out, *curves, **options)


def main() -> None:
    # Start loading PSIM (license checkout) in the background so the first tool call is fast.
    engine.start_loading().add_done_callback(
        lambda f: print(f"PSIM load failed: {f.exception()}" if f.exception() else f"Loaded {engine.version_name}", file=sys.stderr)
    )
    mcp.run("stdio")


if __name__ == "__main__":
    main()
