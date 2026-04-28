"""Environment capture (TESTPLAN §2).

Serializes hardware + software + run metadata to env.json so that all
campaign artifacts are diff-able after the fact. Failures in optional
sub-collections degrade gracefully — a missing tool produces a `null` field,
not a crashed run.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _run(cmd: list[str], timeout: int = 15) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return out.stdout if out.returncode == 0 else (out.stdout + "\n--STDERR--\n" + out.stderr)
    except Exception as e:  # noqa: BLE001
        return f"<error: {e!r}>"


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _utc_now() -> _dt.datetime:
    """Timezone-aware UTC now (datetime.utcnow() is deprecated in 3.12)."""
    return _dt.datetime.now(_dt.timezone.utc)


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # noqa: WPS433

        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_hip_version"] = getattr(torch.version, "hip", None)
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        info["torch_git_sha"] = getattr(torch.version, "git_version", None)
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            try:
                info["device_props_0"] = {
                    k: getattr(torch.cuda.get_device_properties(0), k)
                    for k in ("name", "major", "minor", "total_memory", "multi_processor_count")
                }
            except Exception as e:  # noqa: BLE001
                info["device_props_0_error"] = repr(e)
    except Exception as e:  # noqa: BLE001
        info["torch_import_error"] = repr(e)
    return info


def _try_import_version(mod: str) -> Optional[str]:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None) or getattr(m, "VERSION", None)
    except Exception:  # noqa: BLE001
        return None


def _sdpa_backend_probe() -> Dict[str, Any]:
    """Best-effort probe of which SDPA backend torch will pick on this device."""
    out: Dict[str, Any] = {}
    try:
        import torch  # noqa: WPS433

        if not torch.cuda.is_available():
            return {"available": False}
        # Backend availability flags (newer torch only).
        be = getattr(torch.backends.cuda, "sdp_kernel", None)
        out["sdp_kernel_ctx_present"] = be is not None
        for flag in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
            f = getattr(torch.backends.cuda, flag, None)
            if callable(f):
                try:
                    out[flag] = bool(f())
                except Exception as e:  # noqa: BLE001
                    out[flag] = repr(e)
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def collect_env(campaign_id: Optional[str] = None) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "run": {
            "campaign_id": campaign_id or _utc_now().strftime("%Y%m%d-%H%M%S"),
            "git_sha": _git_sha(),
            "host": socket.gethostname(),
            "user": os.environ.get("USER"),
            "timestamp_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "cwd": str(Path.cwd()),
            "env_vars_relevant": {
                k: os.environ.get(k)
                for k in (
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "PYTORCH_HIP_ALLOC_CONF",
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "TORCH_LOGS",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "HSA_FORCE_FINE_GRAIN_PCIE",
                    "MIOPEN_FIND_MODE",
                    "AITER_*",
                )
            },
        },
        "hardware": {
            "rocm_smi_dump": _run(["rocm-smi", "-a", "--showpower", "--showclocks", "--showtemp", "--showmeminfo", "vram"]),
            "rocminfo": _run(["rocminfo"]),
            "lscpu": _run(["lscpu"]),
            "nproc": _run(["nproc"]),
            "free_h": _run(["free", "-h"]),
        },
        "software": {
            "rocm_version_file": _read_if_exists("/opt/rocm/.info/version") or _read_if_exists("/opt/rocm/share/.info/version"),
            "hipconfig": _run(["hipconfig", "--full"]),
            "hipcc_version": _run(["hipcc", "--version"]),
            "rocm_agent_enumerator": _run(["rocm_agent_enumerator"]),
            "torch": _torch_info(),
            "torchvision_version": _try_import_version("torchvision"),
            "triton_version": _try_import_version("triton"),
            "aiter_version": _try_import_version("aiter"),
            "flash_attn_version": _try_import_version("flash_attn"),
            "rccl_version": _run(["bash", "-lc", "strings $(ldconfig -p | awk '/librccl.so/{print $4; exit}') 2>/dev/null | grep -E 'NCCL|RCCL' | head -n 5"]),
            "miopen_version_file": _read_if_exists("/opt/rocm/include/miopen/version.h"),
            "sdpa_probe": _sdpa_backend_probe(),
        },
    }
    return env


def _read_if_exists(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text().strip()
    except Exception:  # noqa: BLE001
        return None


def write_env(out_dir: Path, campaign_id: Optional[str] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env(campaign_id=campaign_id)
    p = out_dir / "env.json"
    p.write_text(json.dumps(env, indent=2, default=str))
    return p


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--campaign-id", default=None)
    args = ap.parse_args()
    p = write_env(args.out, args.campaign_id)
    print(f"wrote {p}")
