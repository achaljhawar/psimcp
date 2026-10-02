"""Thread-confined wrapper around Altair PSIM's Python API (Psim_Class.py / psim2.dll).

All DLL calls run on one dedicated worker thread. Loading psim2.dll checks out a
license and takes tens of seconds, so the PSIM object is created once and reused.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import glob
import io
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def find_psim_folder() -> str:
    """Resolve the PSIM install folder: $PSIM_PATH, else the newest C:\\Altair\\Altair_PSIM_* with psim2.dll."""
    env = os.environ.get("PSIM_PATH")
    if env:
        return env
    candidates = sorted(
        (d for d in glob.glob(r"C:\Altair\Altair_PSIM_*") if os.path.isfile(os.path.join(d, "psim2.dll"))),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\POWERSIM\Folders64") as key:
                return winreg.QueryValueEx(key, "LASTPATH")[0]
        except OSError:
            pass
    raise RuntimeError("PSIM installation not found. Set the PSIM_PATH environment variable to the PSIM folder.")


class PsimError(RuntimeError):
    pass


class PsimEngine:
    def __init__(self, psim_folder: str | None = None):
        self.folder = psim_folder or find_psim_folder()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="psim")
        self._load_future: concurrent.futures.Future | None = None
        self._lock = threading.Lock()
        self.psim: Any = None
        self.mod: Any = None  # the Psim_Class module
        self.version_name = ""

    # ---- lifecycle -------------------------------------------------------

    def start_loading(self) -> concurrent.futures.Future:
        with self._lock:
            if self._load_future is None:
                self._load_future = self._executor.submit(self._load)
            return self._load_future

    def _load(self) -> None:
        src = Path(self.folder) / "Python" / "Source" / "psimapipy-sources"
        if not (src / "Psim_Class.py").is_file():
            raise PsimError(f"Psim_Class.py not found under {src}. Is PSIM_PATH ({self.folder}) a PSIM 2026+ install?")
        sys.path.insert(0, str(src))
        import Psim_Class  # type: ignore[import-not-found]

        def log(msg: bytes, msg_type: int, thread_id: int) -> None:
            text = msg.decode("utf-8", errors="replace") if msg else ""
            print(f"[psim:{thread_id} type={msg_type}] {text}", file=sys.stderr)

        with contextlib.redirect_stdout(sys.stderr):
            psim = Psim_Class.PSIM(self.folder, log)
        if not psim.IsValid():
            raise PsimError(f"Failed to load PSIM from {self.folder} (license unavailable or psim2.dll failed to init).")
        self.mod = Psim_Class
        self.psim = psim
        self.version_name = psim.get_psim_version_name()

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, str]:
        """Run fn(psim, ...) on the PSIM thread. Returns (value, text PSIM printed while running)."""
        self.start_loading().result()

        def run() -> tuple[T, str]:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                value = fn(self.psim, *args, **kwargs)
            return value, buf.getvalue().strip()

        return self._executor.submit(run).result()

    # ---- helpers used inside call() --------------------------------------

    def open_schematic(self, path: str) -> Any:
        sch = self.psim.PsimFileOpen(path)
        if not sch:
            raise PsimError(f"Unable to open schematic: {path}")
        return sch
