# -*- coding: utf-8 -*-
"""在开启自检重试的前提下，比较几组 P/码率配置。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import core  # noqa: E402

import numpy as np  # noqa: E402

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CONFIGS = [(128, "conv13"), (112, "conv13"), (160, "conv14"), (144, "conv13")]

wav, info = core.read_audio_bytes((APP / "assets" / "demo.wav").read_bytes())
rng = np.random.default_rng(900913)
secrets = []
while len(secrets) < TRIALS:
    text = f"{int(rng.integers(0, 0x10000)):04X}"
    if text not in secrets:
        secrets.append(text)

rows = []
for p, scheme in CONFIGS:
    core.HPP_MAX_TOKENS = p
    core.HPP_SCHEME = scheme
    exact, raws, tries, stois = 0, [], [], []
    for text in secrets:
        result = core.analyze_embedding(wav, "hpp", core.hex_to_bits(text), "demo-2026")
        exact += int(result["exact"])
        raws.append(float(result["raw_bit_accuracy"]))
        tries.append(int(result["attempts"]))
        stois.append(float(result["stoi"]))
    row = {"p": p, "scheme": scheme, "exact_rate": exact / len(secrets),
           "raw_mean": float(np.mean(raws)), "tries_mean": float(np.mean(tries)),
           "stoi": float(np.mean(stois))}
    rows.append(row)
    print(f"P={p:<4} {scheme}  exact={row['exact_rate']*100:5.1f}%  raw={row['raw_mean']:.3f}  "
          f"tries={row['tries_mean']:.2f}  stoi={row['stoi']:.3f}", flush=True)

(APP / "runtime" / "config_with_retry.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("written config_with_retry.json")
