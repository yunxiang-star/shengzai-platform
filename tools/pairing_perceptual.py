# -*- coding: utf-8 -*-
"""配对策略 × 感知失真 × 重编码存活率的权衡扫描。

动机：用户听出明显失真。检查发现 direct 的 SNR 是负的（-7 dB），
因为 partner 映射是随机配对，替换码字可能落在码本里很远的格子。
这里把“配对距离”变成一个可调旋钮：
  random   —— 当前实现（随机配对）
  nearest  —— 用贪心最近邻做完美匹配，尽量让每次替换的码本距离最小
  mid      —— 从距离排序表里跳过最近 30% 的边再匹配（折中）
每个策略都在同一段音频上测：完整恢复率、原始比特存活率、STOI、LSD、分段 SNR、掩码超阈比例。
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

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SKIP_FRACTION = {"random": None, "nearest": 0.0, "mid": 0.30}


def build_matching(kind: str):
    if kind == "random":
        return core.build_pairs()
    codebook = core.load_model().quantizer.layers[core.LAYER].codebook.embed.squeeze().float().cpu()
    distance = torch.cdist(codebook, codebook).numpy()
    n = distance.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    weights = distance[iu, ju]
    order = np.argsort(weights)
    skip = SKIP_FRACTION.get(kind, 0.0) or 0.0
    start = int(len(order) * skip)
    used = np.zeros(n, dtype=bool)
    pairs = []
    for index in order[start:]:
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
    wav, info = core.read_audio_bytes((APP / "assets" / "demo.wav").read_bytes())
    message = core.hex_to_bits("C0DE")
    original_make_maps = core.make_maps
    rows = []

    for kind in ("random", "mid", "nearest"):
        pairs = build_matching(kind)
        core.make_maps = make_maps_factory(pairs)
        exact, raws, stois, lsds, segsnrs, audit = 0, [], [], [], [], []
        for _ in range(TRIALS):
            result = core.analyze_embedding(wav, "direct", message, "demo-2026", nonce=f"pair-{kind}")
            exact += int(result["exact"])
            raws.append(float(result["raw_bit_accuracy"]))
            stois.append(float(result["stoi"]))
            ref, test = result["cover_wave"], result["stego_wave"]
            n = min(len(ref), len(test))
            ref, test = ref[:n], test[:n]
            lsds.append(pc.lsd_db(ref, test))
            segsnrs.append(pc.seg_snr_db(ref, test)["mean"])
            audit.append(pc.nmr(ref, test)["offset_24db"]["audible_fraction"])
        row = {
            "pairing": kind,
            "trials": TRIALS,
            "exact_rate": exact / TRIALS,
            "raw_mean": float(np.mean(raws)),
            "stoi": float(np.mean(stois)),
            "lsd_db": float(np.mean(lsds)),
            "seg_snr_db": float(np.mean(segsnrs)),
            "audible_fraction_24db": float(np.mean(audit)),
        }
        rows.append(row)
        print(f"{kind:8s} exact={row['exact_rate']*100:5.1f}% raw={row['raw_mean']:.3f} "
              f"stoi={row['stoi']:.3f} lsd={row['lsd_db']:.2f}dB segSNR={row['seg_snr_db']:.2f}dB "
              f"audible={row['audible_fraction_24db']*100:.1f}%", flush=True)

    core.make_maps = original_make_maps
    out = APP / "runtime" / "pairing_perceptual.json"
    out.write_text(json.dumps({"clip": "demo.wav", "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
