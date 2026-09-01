#!/usr/bin/env python3
"""Run lightweight checks that do not require model weights or source databases."""

from __future__ import annotations

import hashlib
from pathlib import Path

from epicatanexus.config import load_config
from epicatanexus.metrics import ndcg_at_k, regression_metrics


FIGURE_SHA256 = {
    "Fig1.png": "683b12ea179bf24dbc5a02f4f0a3e41337c066fe361a21ee6710649c0a60559f",
    "Fig2.png": "91b475ea18e80701944b022adc2ca5b7869443fce81fbc86ecdec3c8e7ec5285",
    "Fig3.png": "3721465421bd7becc7d631088fc307d6493240ac7c40a36cca243ee31af90190",
    "Fig4.png": "b0c1ca0898ff6ca38757ac994f68bf9e83106249cfb09ea74b94a431c6a9e002",
    "Fig5.png": "2fa589cf21361e0e1f58966c0f342f22af3d690b5fbbd7d50a717bbb8455bee8",
    "Fig6.png": "f74ac0e594c72809589375da27869dc92de079f79ca393bd7611f5fd7d465ea5",
    "Fig7.png": "eecadf42fc168399ddec01caf42b0bf24dba02db30a56915f59b7b2466936df6",
}


WEB_FIGURE_SHA256 = {
    "Fig1.webp": "874c0441c3331c739c8cdcee30e1dec209c562bf36d85b6721c167b270ba23f5",
    "Fig2.webp": "a15469bd22bffccafc1e0ec2a0719aa6d13f1d3447e3f598bdab940cd06c451b",
    "Fig3.webp": "8779987a727f4169b2b139dab9c21c8782d49a053d709d488d48c3c52f96fcbe",
    "Fig4.webp": "a1ce38c975b432f425ffc040816857c658c6e44025682d6582d3b97732ebf009",
    "Fig5.webp": "841b79ec558e4c9c5a5a15f49e46ba0930429e855dbb427374e5d851bb0df0bd",
    "Fig6.webp": "c0006a6010dfdc46f399b79a7eb813d9765c07204860a9732d063f97dd3d3d6c",
    "Fig7.webp": "75152c57ee58c0b01fc3f70760441a77f1caa5b1dcdc0492efce24fa2f406241",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "configs/kcat.yaml")["seed"] == 3407
    assert regression_metrics([0, 1, 2], [0, 1, 2])["r2"] == 1.0
    assert ndcg_at_k([1, 0, 0], 3) == 1.0
    for filename, expected in FIGURE_SHA256.items():
        path = root / "assets/figures" / filename
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"Figure checksum mismatch: {filename}"
    for filename, expected in WEB_FIGURE_SHA256.items():
        path = root / "assets/figures/web" / filename
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"Web figure checksum mismatch: {filename}"
    print("Smoke test passed: configs, metrics, and original/web figures are valid.")


if __name__ == "__main__":
    main()

