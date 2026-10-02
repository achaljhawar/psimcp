# psim-mcp

An MCP server for **Altair PSIM 2026**. It lets Claude (or any MCP client) inspect, edit, build and simulate
power-electronics schematics, run parameter sweeps, and analyze the resulting waveforms.

It wraps the PSIM Python API that ships with PSIM (`<PSIM>\Python\Source\psimapipy-sources\Psim_Class.py`),
loading it straight from your install, so nothing proprietary is vendored here.

## Requirements

- Windows with a licensed **PSIM 2026.x** install (tested with 2026.1.0.8096)
- [uv](https://docs.astral.sh/uv/) (Python 3.11+)

## Tools

| Tool | What it does |
|---|---|
| `psim_status` | Loads PSIM; reports version, install and output folders |
| `find_examples` | Searches the PSIM install for example `.psimsch` files |
| `inspect_schematic` | Elements and parameters, output probes, **undefined variables** |
| `schematic_to_python` | Full topology as a PSIM build script (coordinates, wires) |
| `simulate` | Transient sim (PSIM / HyperSpice / LTspice) with variable overrides and in-memory element edits; returns a `result_id` and steady-state summary per curve |
| `parameter_sweep` | Parallel grid or zipped sweeps, tabulating steady-state mean and ripple per run |
| `analyze_waveform` | Mean, RMS, AC RMS, min/max, peak-to-peak, fundamental frequency over a window |
| `get_waveform` | Downsampled samples for plotting or close inspection |
| `measure_cursors` | SIMVIEW-style two-cursor readout: values at each cursor, delta, 1/dx, and between them avg, RMS, avg\|x\|, min/max; optional P, S, PF and THD |
| `fft_analysis` | Harmonic spectrum over whole fundamental cycles: DC, harmonic amplitudes and phases, THD; can open the same window in SIMVIEW's FFT view |
| `list_results`, `load_graph_file`, `export_csv` | Result management |
| `open_in_simview` | Open results in SIMVIEW, optionally in FFT view, log axes, or trimmed to an x window |
| `edit_schematic` | Change element values or enable/disable elements, then save |
| `create_schematic` | Build a new schematic from elements and wires |
| `design_suite_get_parameters`, `design_suite_update_parameters` | PSIM Design Suite parameters |
| `simcoder_generate_code` | SimCoder embedded C code generation |

Loading PSIM checks out a license and takes about 30 s. The server starts loading it in the background at
startup, so it is usually ready by the first tool call.

## Setup

```powershell
cd C:\Users\Achal\projects\psimcp
uv sync
```

### Claude Code

This repo includes `.mcp.json`, so Claude Code started in this folder picks the server up automatically.
To make it available everywhere:

```powershell
claude mcp add psim --scope user -- uv --directory C:\Users\Achal\projects\psimcp run psim-mcp
```

### Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "psim": {
      "command": "uv",
      "args": ["--directory", "C:\\Users\\Achal\\projects\\psimcp", "run", "psim-mcp"]
    }
  }
}
```

### Configuration (environment variables)

| Variable | Default |
|---|---|
| `PSIM_PATH` | Newest `C:\Altair\Altair_PSIM_*` containing `psim2.dll`, then the registry `LASTPATH` |
| `PSIM_MCP_OUTPUT_DIR` | `%USERPROFILE%\psim_mcp_output` (graph files, CSV exports) |

## Example prompts

- "Inspect `buck.psimsch`, then simulate it with Vin=48, L=50u, C=100u, R=2.5, Fsw=200k and tell me the output ripple."
- "Sweep L over 20u, 50u and 100u and Fsw over 100k and 200k. Which combination keeps inductor ripple under 1 A?"
- "Find the LLC resonant example and explain its topology."

## Testing

```powershell
uv run python tests/smoke_test.py
uv run python tests/test_analysis.py   # FFT/cursor math, no PSIM needed
```

This runs the server over stdio against PSIM's bundled buck converter sample and checks inspect, simulate,
element edits, sweep, export, reload and edit/save.

## Notes

- Simulation results are cached in memory (the last 200) and written to graph files in the output folder.
  `load_graph_file` re-imports any graph file.
- `simulate` `element_edits` apply only to that run. `edit_schematic` writes a file.
- `design_suite_update_parameters` changes the design's files on disk.
- SIMVIEW has no API for its measure cursors, so `measure_cursors` computes the same readout from the data.
  SIMVIEW's FFT can be opened directly (`FFT=1`). SIMVIEW transforms everything on screen, so
  `fft_analysis(open_simview=True)` and `open_in_simview(x_start, x_end)` write a trimmed copy of the window.
- `fft_analysis` amplitudes are peak values, like SIMVIEW. Pass `fundamental_hz` when you know it. Otherwise it is
  estimated from the trailing half of the run, which needs several cycles there.
