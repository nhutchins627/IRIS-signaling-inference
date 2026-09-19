"""Provenance capture: record how each output was produced.

Every figure script writes a sidecar JSON next to its outputs recording the
package versions, git commit, hostname, config, and input files used. This is
what makes a regenerated panel auditable a year later.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

_PACKAGES = ["numpy", "pandas", "scanpy", "anndata", "scvi", "sklearn",
             "torch", "scipy", "matplotlib", "seaborn", "gseapy"]


def package_versions() -> Dict[str, str]:
    out = {}
    for name in _PACKAGES:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "not installed"
    return out


def git_commit(path: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path or Path(__file__).parent),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "not a git repository"


def file_fingerprint(path: str | Path, hash_bytes: int = 1 << 20) -> Dict[str, Any]:
    """Size, mtime, and a partial content hash.

    Only the first ``hash_bytes`` are hashed: the atlases run to tens of GB and
    a full digest would dominate runtime. Size + mtime + head hash is enough to
    detect a swapped or truncated input.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read(hash_bytes))
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256_head": h.hexdigest(),
    }


def gpu_info() -> Dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        }
    except Exception:
        return {"cuda_available": False, "note": "torch not installed"}


def record(name: str, outdir: str | Path, *, inputs: Iterable[str | Path] = (),
           params: Dict[str, Any] | None = None,
           results: Dict[str, Any] | None = None) -> Path:
    """Write ``<outdir>/<name>.provenance.json`` and return its path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "git_commit": git_commit(),
        "packages": package_versions(),
        "gpu": gpu_info(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": [file_fingerprint(p) for p in inputs],
        "parameters": params or {},
        "results": results or {},
    }
    path = outdir / f"{name}.provenance.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def check_environment(strict: bool = False) -> Dict[str, Any]:
    """Compare the live environment against the versions used for the paper."""
    expected = {
        "scvi": "0.20.3", "scanpy": "1.10.0", "anndata": "0.10.6",
        "torch": "2.2.2+cu121", "sklearn": "1.4.1.post1", "scipy": "1.12.0",
    }
    actual = package_versions()
    mismatches = {
        k: {"expected": v, "actual": actual.get(k)}
        for k, v in expected.items()
        if actual.get(k) not in (v, "not installed")
    }
    missing = [k for k in expected if actual.get(k) == "not installed"]
    report = {"actual": actual, "mismatches": mismatches, "missing": missing}
    if mismatches or missing:
        msg = ("Environment differs from the one used for the published "
               f"figures.\n  mismatched: {mismatches}\n  missing: {missing}")
        if strict:
            raise RuntimeError(msg)
        print("WARNING:", msg, file=sys.stderr)
    return report
