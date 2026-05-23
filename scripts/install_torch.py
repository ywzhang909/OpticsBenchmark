#!/usr/bin/env python3
"""
Auto-detect CUDA and install the matching PyTorch variant.

Uses uv's built-in ``--torch-backend=auto`` to automatically select
the correct PyTorch wheel index based on the detected CUDA driver version.

Reference: https://uv.doczh.com/guides/integration/pytorch/#_3

Usage:
    uv run python scripts/install_torch.py

Or directly:
    UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio --upgrade
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def detect_cuda_from_smi() -> tuple[int, int] | None:
    """Parse CUDA Version from ``nvidia-smi`` output."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass
    return None


# Known ``--torch-backend`` values supported by uv 0.6.12
_UV_KNOWN_BACKENDS: set[str] = {
    "auto", "cpu", "cu126", "cu125", "cu124", "cu123", "cu122",
    "cu121", "cu120", "cu118", "cu117", "cu116", "cu115", "cu114",
    "cu113", "cu112", "cu111", "cu110", "cu102", "cu101", "cu100",
    "cu92", "cu91", "cu90", "cu80",
}

_PYTORCH_INDEX_BASE = "https://download.pytorch.org/whl"


def _lookup_backend(cuda_ver: tuple[int, int] | None) -> str:
    """Map CUDA version to a PyTorch backend tag (e.g. ``cu130``)."""
    if cuda_ver is None:
        return "cpu"
    table: list[tuple[tuple[int, int], str]] = [
        ((13, 0), "cu130"),
        ((12, 8), "cu128"),
        ((12, 6), "cu126"),
        ((12, 4), "cu124"),
        ((12, 1), "cu121"),
        ((11, 8), "cu118"),
    ]
    for min_ver, tag in table:
        if cuda_ver >= min_ver:
            return tag
    return "cpu"


def _backend_is_supported(backend: str) -> bool:
    """Check if uv's ``--torch-backend`` knows this value."""
    return backend in _UV_KNOWN_BACKENDS


def build_install_cmd(backend: str) -> list[str]:
    """Build the install command, falling back to ``--index-url`` for
    CUDA versions newer than what the installed uv knows about."""
    pkgs = ["torch", "torchvision", "torchaudio"]

    if _backend_is_supported(backend):
        return [
            "uv", "pip", "install", *pkgs,
            f"--torch-backend={backend}",
            "--upgrade",
        ]

    index_url = f"{_PYTORCH_INDEX_BASE}/{backend}"
    tag = backend.lstrip("cu")  # e.g. "130"
    # Pin to the latest build on that index using local-version hint
    return [
        "uv", "pip", "install", *pkgs,
        "--index-url", index_url,
        f"--only-binary=:all:",
        "--upgrade",
    ]


def update_pyproject_sources(backend: str) -> bool:
    """Update ``[tool.uv.sources]`` in ``pyproject.toml`` so future ``uv sync``
    uses the correct PyTorch index."""
    root = Path(__file__).resolve().parent.parent
    pp = root / "pyproject.toml"
    if not pp.exists():
        print("  [WARN] pyproject.toml not found, skipping source update")
        return False

    text = pp.read_text(encoding="utf-8")
    index_name = f"pytorch-{backend}" if backend != "cpu" else "pytorch-cpu"

    # Replace the sources block (lines may or may not have leading whitespace)
    pattern = (
        r'\[tool\.uv\.sources\]\s*\n'
        r'(?:\s*torch\s*=\s*\[.*?\]\s*\n)?'
        r'(?:\s*torchvision\s*=\s*\[.*?\]\s*\n)?'
        r'(?:\s*torchaudio\s*=\s*\[.*?\]\s*\n)?'
    )
    replacement = (
        f"[tool.uv.sources]\n"
        f'torch = [{{ index = "{index_name}" }}]\n'
        f'torchvision = [{{ index = "{index_name}" }}]\n'
        f'torchaudio = [{{ index = "{index_name}" }}]\n'
    )

    if re.search(pattern, text, re.MULTILINE):
        text = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
    elif "[tool.uv.sources]" not in text:
        # No sources block at all – append before [[tool.uv.index]]
        idx_pos = text.find("[[tool.uv.index]]")
        if idx_pos != -1:
            text = text[:idx_pos] + replacement + "\n" + text[idx_pos:]
        else:
            text += "\n" + replacement

    pp.write_text(text, encoding="utf-8")
    print(f"  [OK] pyproject.toml [tool.uv.sources] → {index_name}")
    return True


def main() -> None:
    print("=" * 60)
    print("  PyTorch CUDA Auto-Install")
    print("  Reference: https://uv.doczh.com/guides/integration/pytorch/#_3")
    print("=" * 60)

    # ── Step 1: Detect CUDA ────────────────────────────────────
    cuda_ver = detect_cuda_from_smi()
    if cuda_ver:
        print(f"\n  [DETECT] CUDA {cuda_ver[0]}.{cuda_ver[1]} (nvidia-smi)")
    else:
        print("\n  [DETECT] No CUDA driver found → CPU-only")

    backend = _lookup_backend(cuda_ver)
    print(f"  [TARGET] PyTorch backend: {backend}")
    if not _backend_is_supported(backend):
        print(f"  [INFO] uv doesn't know '{backend}' yet — using --index-url directly")

    # ── Step 2: Install via uv pip ──────────────────────────────
    cmd = build_install_cmd(backend)
    print(f"\n  [INSTALL] {' '.join(cmd[:4])} ... --upgrade\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n  [FAIL] Installation failed (exit code {result.returncode})")
        sys.exit(1)

    # ── Step 3: Update pyproject.toml sources ───────────────────
    print("\n  [CONFIG] Updating pyproject.toml for future uv sync...")
    update_pyproject_sources(backend)

    # ── Step 4: Verify ──────────────────────────────────────────
    print("\n  [VERIFY] Checking installation...")
    try:
        ver = subprocess.run(
            [
                sys.executable, "-c",
                "import torch; "
                f"print(f'  Torch {{torch.__version__}}'); "
                f"print(f'  CUDA available: {{torch.cuda.is_available()}}'); "
                f"print(f'  CUDA version: {{torch.version.cuda}}'); "
                f"print(f'  Device count: {{torch.cuda.device_count()}}'); "
                f"print(f'  Device name: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}}')",
            ],
            capture_output=True,
            text=True,
        )
        print(ver.stdout)
        if ver.returncode != 0:
            print(ver.stderr)
    except Exception as e:
        print(f"  [WARN] Verification failed: {e}")

    print("=" * 60)
    print("  Done! 🚀")
    print("=" * 60)


if __name__ == "__main__":
    main()
