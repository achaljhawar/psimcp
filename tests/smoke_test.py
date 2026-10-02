"""End-to-end check: launch the server over stdio and exercise the tools on PSIM's bundled buck sample.

Run with:  uv run python tests/smoke_test.py
Needs a licensed PSIM 2026 install.
"""

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from psim_mcp.engine import find_psim_folder

# buck.psimsch leaves these variables undefined; the caller must supply them.
BUCK = {"Vin": 48, "L": "50u", "C": "100u", "R": 2.5, "Fsw": "200k", "TotalTime": 0.005}

SAMPLES = Path(find_psim_folder()) / "Python" / "Samples" / "PythonTest"


async def call(session: ClientSession, name: str, **args):
    res = await session.call_tool(name, args, read_timeout_seconds=600)
    text = "\n".join(c.text for c in res.content if hasattr(c, "text"))
    status = "ERROR" if res.is_error else "ok"
    print(f"\n=== {name} [{status}]\n{text[:1500]}")
    if res.is_error:
        raise SystemExit(f"{name} failed")
    return json.loads(text) if text[:1] in "{[" else text


async def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="psim_mcp_test_"))
    for f in ("buck.psimsch", "parameters-1.txt"):
        shutil.copy(SAMPLES / f, work)
    buck = str(work / "buck.psimsch")

    params = StdioServerParameters(command=sys.executable, args=["-m", "psim_mcp.server"], env={"PSIM_MCP_OUTPUT_DIR": str(work / "out")})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        await call(session, "psim_status")
        insp = await call(session, "inspect_schematic", schematic_path=buck, element_types=["R", "L", "C"])
        assert set(insp["undefined_variables"]) == {"R", "Vin", "L", "Fsw", "C"}
        py = await call(session, "schematic_to_python", schematic_path=buck)
        assert "PsimCreateNewElement" in py

        r = await call(session, "simulate", schematic_path=buck, parameters=BUCK)
        rid = r["result_id"]
        await call(session, "analyze_waveform", result_id=rid, curve="Vout", x_start=0.004)
        await call(session, "get_waveform", result_id=rid, curves=["Vout"], max_points=10)
        cur = await call(session, "measure_cursors", result_id=rid, x1=0.004, x2=0.005, curves=["Vout", "I(R1)"],
                         voltage_curve="Vout", current_curve="I(R1)")
        assert abs(cur["power"]["power_factor"] - 1) < 0.01, "resistive load should have PF 1"
        fft = await call(session, "fft_analysis", result_id=rid, curves=["Vout"], fundamental_hz=200e3, cycles=10)
        assert fft["window"][1] == r["x_range"][1] and "warning" not in fft

        edited = await call(
            session,
            "simulate",
            schematic_path=buck,
            parameters=BUCK,
            element_edits=[{"element_type": "R", "element_name": "R1", "parameter": "Resistance", "value": 5}],
        )
        assert edited["curves"]["I(R1)"]["ss_mean"] < r["curves"]["I(R1)"]["ss_mean"], "edit had no effect"

        sweep = await call(session, "parameter_sweep", schematic_path=buck, sweep={"Vin": [24, 48, 96]}, base_parameters=BUCK, curves=["Vout"])
        assert sweep["succeeded"] == 3

        csv_path = await call(session, "export_csv", result_id=rid, curves=["Vout"])
        loaded = await call(session, "load_graph_file", graph_path=r["graph_file"])
        assert loaded["points"] == r["points"]

        out = str(work / "buck_edited.psimsch")
        await call(session, "edit_schematic", schematic_path=buck, output_path=out,
                   element_edits=[{"element_type": "C", "element_name": "C1", "parameter": "Capacitance", "value": "100u"}])
        insp = await call(session, "inspect_schematic", schematic_path=out, element_types=["C"])
        assert insp["elements"][0]["parameters"]["Capacitance"] == "100u"
        print(f"\nAll checks passed. Artifacts in {work}  (csv: {csv_path})")


if __name__ == "__main__":
    asyncio.run(main())
