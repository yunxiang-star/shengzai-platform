# -*- coding: utf-8 -*-
"""同一 HPP 配置下，比较不同音频的原始比特存活率与消息恢复率。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import core  # noqa: E402

import numpy as np  # noqa: E402

SAMPLES = Path(r"D:/语音深伪与隐写/01_检测实验台/samples/real_ms")
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
core.HPP_MAX_TOKENS = 160
core.HPP_SCHEME = "conv14"

clips = [APP / "assets" / "demo.wav"]
clips += [SAMPLES / n for n in [
    "real_spk3752_14.wav",   # 7.21s
    "real_spk5536_16.wav",   # 3.17s
    "real_spk6319_19.wav",   # 7.29s
    "real_spk7976_22.wav",   # 3.55s
]]

rng = np.random.default_rng(777)
secrets = []
while len(secrets) < TRIALS:
    text = f"{int(rng.integers(0, 0x10000)):04X}"
    if text not in secrets:
        secrets.append(text)

summary = []
for clip in clips:
    if not clip.exists():
        print("missing", clip, flush=True)
        continue
    wav, info = core.read_audio_bytes(clip.read_bytes())
    frames = core.load_model().encode(core._signal_tensor(wav), bandwidth=core.BANDWIDTH).audio_codes.shape[3]
    exact, raws = 0, []
    for text in secrets:
        result = core.analyze_embedding(wav, "hpp", core.hex_to_bits(text), "demo-2026")
        exact += int(result["exact"])
        raws.append(float(result["raw_bit_accuracy"]))
    row = {
        "clip": clip.name,
        "seconds": info["duration_seconds"],
        "frames": int(frames),
        "exact_rate": exact / len(secrets),
        "raw_mean": float(np.mean(raws)),
        "raw_min": float(np.min(raws)),
    }
    summary.append(row)
    print(f"{clip.name:26s} {row['seconds']:5.2f}s f={row['frames']:4d} exact={row['exact_rate']*100:5.1f}% raw={row['raw_mean']:.3f} min={row['raw_min']:.3f}", flush=True)

(APP / "runtime" / "clip_compare.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("written clip_compare.json")
