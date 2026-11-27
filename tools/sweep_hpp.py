# -*- coding: utf-8 -*-
"""扫描 HPP 的 P 与码率，找出短音频上真正稳的配置。

只改 core 的模块级参数（HPP_MAX_TOKENS / HPP_SCHEME），不改算法本身。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import core  # noqa: E402

import numpy as np  # noqa: E402

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CLIP = sys.argv[2] if len(sys.argv) > 2 else str(APP / "assets" / "demo.wav")

wav, info = core.read_audio_bytes(Path(CLIP).read_bytes())
frames = core.load_model().encode(core._signal_tensor(wav), bandwidth=core.BANDWIDTH).audio_codes.shape[3]
print(f"clip={Path(CLIP).name} {info['duration_seconds']}s frames={frames}", flush=True)

rng = np.random.default_rng(424242)
secrets = []
while len(secrets) < TRIALS:
    text = f"{int(rng.integers(0, 0x10000)):04X}"
    if text not in secrets:
        secrets.append(text)

configs = []
for p in (96, 112, 128, 144, 160):
    for scheme in ("conv13", "conv14"):
        needed = (core.PAYLOAD_BITS + 6) * int(scheme[-1])
        if p < needed:
            continue
        configs.append((p, scheme))

results = []
for p, scheme in configs:
    core.HPP_MAX_TOKENS = p
    core.HPP_SCHEME = scheme
    exact = 0
    raws = []
    infeasible = 0
    for text in secrets:
        message = core.hex_to_bits(text)
        try:
            result = core.analyze_embedding(wav, "hpp", message, "demo-2026")
        except core.EmbeddingInfeasible:
            infeasible += 1
            continue
        exact += int(result["exact"])
        raws.append(float(result["raw_bit_accuracy"]))
    rate = exact / max(len(secrets) - infeasible, 1)
    row = {
        "p": p,
        "scheme": scheme,
        "carrier_bits": (core.PAYLOAD_BITS + 6) * int(scheme[-1]),
        "trials": len(secrets),
        "infeasible": infeasible,
        "exact_rate": rate,
        "raw_mean": float(np.mean(raws)) if raws else None,
    }
    results.append(row)
    raw_text = f"{row['raw_mean']:.3f}" if row["raw_mean"] is not None else "  n/a"
    print(f"P={p:<4} {scheme}  exact={rate*100:5.1f}%  raw={raw_text}  infeasible={infeasible}", flush=True)

out = APP / "runtime" / "sweep_hpp.json"
out.write_text(json.dumps({"clip": Path(CLIP).name, "frames": int(frames), "results": results},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print("written", out)
