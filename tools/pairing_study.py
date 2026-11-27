# -*- coding: utf-8 -*-
"""在 direct 与 hpp 两种模式下，对比随机配对与最近邻配对的音质/恢复率。

大样本版：每格 20 次，另外补一段独立音频，避免只在示例音频上成立。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / "tools"))
import core  # noqa: E402
import perceptual as pc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CLIPS = [
    APP / "assets" / "demo.wav",
    Path(r"D:/语音深伪与隐写/01_检测实验台/samples/real_ms/real_spk3752_14.wav"),
]


def nearest_pairs():
    codebook = core.load_model().quantizer.layers[core.LAYER].codebook.embed.squeeze().float().cpu()
    distance = torch.cdist(codebook, codebook).numpy()
    n = distance.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(distance[iu, ju])
    used = np.zeros(n, dtype=bool)
    pairs = []
    for index in order:
        a, b = int(iu[index]), int(ju[index])
        if used[a] or used[b]:
            continue
        used[a] = used[b] = True
        pairs.append((a, b))
        if len(pairs) == n // 2:
            break
    return pairs


def make_maps_factory(pairs):
    def make_maps(key):
        rng = np.random.default_rng(core.key_seed(key, "codebook"))
        bit_of = np.full(1024, -1, dtype=np.int8)
        partner = np.arange(1024, dtype=np.int16)
        for a, b in pairs:
            if rng.integers(0, 2):
                a, b = b, a
            bit_of[a], bit_of[b] = 0, 1
            partner[a], partner[b] = b, a
        return bit_of, partner
    return make_maps


def main():
    original = core.make_maps
    random_pairs = core.build_pairs()
    near_pairs = nearest_pairs()
    rows = []

    for clip in CLIPS:
        if not clip.exists():
            print("skip missing", clip)
            continue
        wav, info = core.read_audio_bytes(clip.read_bytes())
        message = core.hex_to_bits("C0DE")
        for mode in ("direct", "hpp"):
            for name, pairs in (("random", random_pairs), ("nearest", near_pairs)):
                core.make_maps = make_maps_factory(pairs)
                exact, raws, stois, lsds, segs, aud, tries = 0, [], [], [], [], [], []
                for i in range(TRIALS):
                    result = core.analyze_embedding(wav, mode, message, "demo-2026",
                                                    nonce=f"{clip.stem}-{mode}-{name}-{i}")
                    exact += int(result["exact"])
                    raws.append(float(result["raw_bit_accuracy"]))
                    stois.append(float(result["stoi"]))
                    tries.append(int(result["attempts"]))
                    ref, test = result["cover_wave"], result["stego_wave"]
                    n = min(len(ref), len(test))
                    lsds.append(pc.lsd_db(ref[:n], test[:n]))
                    segs.append(pc.seg_snr_db(ref[:n], test[:n])["mean"])
                    aud.append(pc.nmr(ref[:n], test[:n])["offset_24db"]["audible_fraction"])
                row = {
                    "clip": clip.name, "mode": mode, "pairing": name, "trials": TRIALS,
                    "exact_rate": exact / TRIALS, "raw_mean": float(np.mean(raws)),
                    "stoi": float(np.mean(stois)), "lsd_db": float(np.mean(lsds)),
                    "seg_snr_db": float(np.mean(segs)),
                    "audible_24db": float(np.mean(aud)),
                    "tries_mean": float(np.mean(tries)),
                }
                rows.append(row)
                print(f"{clip.stem[:18]:18s} {mode:6s} {name:7s} exact={row['exact_rate']*100:5.1f}% "
                      f"raw={row['raw_mean']:.3f} stoi={row['stoi']:.3f} lsd={row['lsd_db']:5.2f}dB "
                      f"segSNR={row['seg_snr_db']:5.2f}dB audible={row['audible_24db']*100:4.1f}%",
                      flush=True)

    core.make_maps = original
    out = APP / "runtime" / "pairing_study.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
