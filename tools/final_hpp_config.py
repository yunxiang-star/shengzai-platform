# -*- coding: utf-8 -*-
"""最终确认：P=128/conv13 + 自检重试，在一批新口令上测成功率。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import core  # noqa: E402

import numpy as np  # noqa: E402

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CLIP = Path(sys.argv[2]) if len(sys.argv) > 2 else APP / "assets" / "demo.wav"

core.HPP_MAX_TOKENS = 128
core.HPP_SCHEME = "conv13"
core.HPP_MAX_ATTEMPTS = 12

wav, info = core.read_audio_bytes(CLIP.read_bytes())
rng = np.random.default_rng(555777)   # 与之前所有实验不同的一批口令
secrets = []
while len(secrets) < TRIALS:
    text = f"{int(rng.integers(0, 0x10000)):04X}"
    if text not in secrets:
        secrets.append(text)

exact, raws, tries, stois, fails = 0, [], [], [], []
for text in secrets:
    result = core.analyze_embedding(wav, "hpp", core.hex_to_bits(text), "demo-2026")
    exact += int(result["exact"])
    raws.append(float(result["raw_bit_accuracy"]))
    tries.append(int(result["attempts"]))
    stois.append(float(result["stoi"]))
    if not result["exact"]:
        fails.append({"secret": text, "recovered": result["recovered_hex"],
                      "raw": float(result["raw_bit_accuracy"]), "tries": int(result["attempts"])})
    print(f"{text} -> {result['recovered_hex']} exact={result['exact']} raw={result['raw_bit_accuracy']:.3f} "
          f"tries={result['attempts']} stoi={result['stoi']:.3f}", flush=True)

summary = {
    "clip": CLIP.name,
    "seconds": info["duration_seconds"],
    "config": "P=128 conv13 attempts<=12",
    "n": len(secrets),
    "exact_rate": exact / len(secrets),
    "raw_mean": float(np.mean(raws)),
    "raw_min": float(np.min(raws)),
    "tries_mean": float(np.mean(tries)),
    "tries_hist": {str(k): int(sum(1 for t in tries if t == k)) for k in sorted(set(tries))},
    "stoi_mean": float(np.mean(stois)),
    "stoi_min": float(np.min(stois)),
    "fails": fails,
}
print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
(APP / "runtime" / "final_hpp_config.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("written final_hpp_config.json")
