# -*- coding: utf-8 -*-
"""量化示例音频在两种模式下的端到端失败率（不改任何算法，只测量）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import core  # noqa: E402

import numpy as np  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODES = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["direct", "hpp"])

wav, info = core.read_audio_bytes((APP / "assets" / "demo.wav").read_bytes())
print(f"audio {info['duration_seconds']}s frames rc={len(wav)}", flush=True)

rng = np.random.default_rng(20261028)
secrets = []
while len(secrets) < N:
    value = int(rng.integers(0, 0x10000))
    text = f"{value:04X}"
    if text not in secrets:
        secrets.append(text)

summary = {}
for mode in MODES:
    rows = []
    for text in secrets:
        message = core.hex_to_bits(text)
        result = core.analyze_embedding(wav, mode, message, "demo-2026")
        rows.append({
            "secret": text,
            "recovered": result["recovered_hex"],
            "exact": bool(result["exact"]),
            "raw": float(result["raw_bit_accuracy"]),
            "attempts": int(result.get("attempts", 1)),
            "stoi": float(result["stoi"]),
        })
        print(f"{mode} {text} -> {result['recovered_hex']} exact={result['exact']} "
              f"raw={result['raw_bit_accuracy']:.3f} tries={result.get('attempts', 1)}", flush=True)
    exact_rate = float(np.mean([r["exact"] for r in rows]))
    raws = [r["raw"] for r in rows]
    summary[mode] = {
        "n": len(rows),
        "exact_rate": exact_rate,
        "raw_mean": float(np.mean(raws)),
        "raw_min": float(np.min(raws)),
        "raw_p10": float(np.percentile(raws, 10)),
        "attempts_mean": float(np.mean([r["attempts"] for r in rows])),
        "attempts_hist": {str(k): int(sum(1 for r in rows if r["attempts"] == k))
                          for k in sorted({r["attempts"] for r in rows})},
        "fails": [r for r in rows if not r["exact"]],
    }
    print(f"\n== {mode}: exact {exact_rate*100:.1f}% over {len(rows)}; raw mean {np.mean(raws):.3f} "
          f"min {np.min(raws):.3f}; attempts mean {summary[mode]['attempts_mean']:.2f} "
          f"hist {summary[mode]['attempts_hist']}", flush=True)
    print(flush=True)

out = APP / "runtime" / f"failure_rate_{N}.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("written", out)
