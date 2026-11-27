# -*- coding: utf-8 -*-
"""扫时间项权重：交换越“就近”，重编码后是否更容易存活。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import core  # noqa: E402

import numpy as np  # noqa: E402

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WEIGHTS = [2.0, 8.0, 24.0, 48.0, 96.0, 240.0]

wav, info = core.read_audio_bytes((APP / "assets" / "demo.wav").read_bytes())
core.HPP_MAX_ATTEMPTS = 1  # 先看单次能力，排除重试干扰

rng = np.random.default_rng(31415)
secrets = []
while len(secrets) < TRIALS:
    text = f"{int(rng.integers(0, 0x10000)):04X}"
    if text not in secrets:
        secrets.append(text)

rows = []
for weight in WEIGHTS:
    core.HPP_TEMPORAL_WEIGHT = weight
    exact, raws, stois = 0, [], []
    for text in secrets:
        result = core.analyze_embedding(wav, "hpp", core.hex_to_bits(text), "demo-2026")
        exact += int(result["exact"])
        raws.append(float(result["raw_bit_accuracy"]))
        stois.append(float(result["stoi"]))
    row = {"weight": weight, "exact_rate": exact / len(secrets),
           "raw_mean": float(np.mean(raws)), "stoi": float(np.mean(stois))}
    rows.append(row)
    print(f"w={weight:6.1f}  exact={row['exact_rate']*100:5.1f}%  raw={row['raw_mean']:.3f}  stoi={row['stoi']:.3f}", flush=True)

(APP / "runtime" / "weight_sweep.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("written weight_sweep.json")
