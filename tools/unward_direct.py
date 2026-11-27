# -*- coding: utf-8 -*-
"""把 UNIWARD 的"按感知代价挑位置"搬到索引域，并用软判决换回精度。

两个假设要验证：
  H1  载波位置不该均匀铺满，应该落在能量高（掩蔽强）的帧上 -> 更好听
  H2  解码端不知道原始码字，但知道"这一对码字隔多远"。
      隔得远 = 即使被 codec 抖动也还是同一边 = 可信；隔得近 = 不可信。
      把这个距离当权重喂给 Viterbi，就是零成本的软判决。

跑法（hinet 环境）：
  PYTHONPATH= python -u tools/unward_direct.py > tools/unward_direct.log 2>&1
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch

import core

SR = core.SR
BLOCK = 4


def frame_energy(model, codes):
    """每帧的总能量（所有 RVQ 层的码字向量平方和）。

    故意用"全部层"而不是只看 L0：嵌入只动 L0，总能量几乎不变，
    这样解码端算出来的排序才和编码端一致。
    """
    total = None
    for layer in range(codes.shape[2]):
        emb = model.quantizer.layers[layer].codebook.embed.squeeze().float()
        token = codes[0, 0, layer].long().cpu()
        energy = (emb[token] ** 2).sum(dim=1)
        total = energy if total is None else total + energy
    return total.numpy()


def select_positions(energy, count, key, strategy):
    frames_total = len(energy)
    if strategy == "uniform":
        return np.linspace(0, frames_total - 1, count, dtype=np.int64)
    blocks = frames_total // BLOCK
    if blocks < 2:
        return np.linspace(0, frames_total - 1, count, dtype=np.int64)
    block_energy = energy[: blocks * BLOCK].reshape(blocks, BLOCK).mean(axis=1)
    level = np.round(10 * np.log10(np.maximum(block_energy, 1e-12)) * 2) / 2  # 0.5 dB 一格
    tie = np.random.default_rng(core.key_seed(key, "block-select")).random(blocks)
    order = np.lexsort((tie, -level))          # 先按能级降序，打平按密钥哈希
    need = min(blocks, int(np.ceil(count / BLOCK)))
    chosen = np.sort(order[:need])
    frames = (chosen[:, None] * BLOCK + np.arange(BLOCK)).reshape(-1)
    return frames[np.linspace(0, len(frames) - 1, count, dtype=np.int64)]


def conv_decode_soft(received, weights, rate, payload_bits):
    generators = core.conv_generators(rate)
    steps = payload_bits + 6
    inf = 1e18
    cost = np.full(64, inf)
    cost[0] = 0.0
    prev_state = np.zeros((steps, 64), dtype=np.int16)
    prev_bit = np.zeros((steps, 64), dtype=np.int8)
    for step in range(steps):
        new = np.full(64, inf)
        lo = step * rate
        obs = received[lo:lo + rate]
        w = weights[lo:lo + rate]
        for state in range(64):
            if cost[state] >= inf:
                continue
            for bit in (0, 1):
                register = (state << 1) | bit
                err = sum(w[i] for i in range(rate) if core.parity(register & generators[i]) != obs[i])
                total = cost[state] + err
                nxt = register & 0x3F
                if total < new[nxt]:
                    new[nxt] = total
                    prev_state[step, nxt] = state
                    prev_bit[step, nxt] = bit
        cost = new
    out = np.zeros(steps, dtype=np.int8)
    state = 0
    for step in range(steps - 1, -1, -1):
        out[step] = prev_bit[step, state]
        state = int(prev_state[step, state])
    return out[:payload_bits]


def pair_weights(tokens, partner, bit_of):
    """解码端可得的可靠度：收到码字与它的配对码字在码本里隔多远。"""
    cb = core.load_model().quantizer.layers[core.LAYER].codebook.embed.squeeze().float().cpu()
    idx = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
    mate = torch.from_numpy(partner[idx.numpy()].astype(np.int64))
    distance = (cb[idx] - cb[mate]).norm(dim=1).numpy()
    scale = float(np.median(distance)) or 1.0
    return np.clip(distance / scale, 0.25, 4.0)


def main():
    model = core.load_model()
    clips = sorted(Path(ROOT, "assets").glob("demo*.wav"))
    clips = [p for p in clips if "backup" not in p.name]
    if not clips:
        raise SystemExit("no demo clip")
    report = {}

    for clip in clips[:1]:
        wav, _ = sf.read(str(clip), dtype="float32")
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            encoded = model.encode(core._signal_tensor(wav), bandwidth=24.0)
            cover_codes = encoded.audio_codes.clone()
            cover_wave = model.decode(cover_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
            cover_pcm, _ = core._pcm16_roundtrip(cover_wave)
            cover_energy = frame_energy(model, cover_codes)
            print(f"[clip] {clip.name}  frames={cover_codes.shape[3]}  "
                  f"energy {10*np.log10(cover_energy.max()):.1f}..{10*np.log10(max(cover_energy.min(),1e-12)):.1f} dB")

        rows = []
        for scheme in ("conv13", "conv12"):
            for strategy in ("uniform", "energy"):
                for key in ("alpha", "bravo", "charlie", "delta"):
                    message = np.random.default_rng(core.key_seed(key, "msg")).integers(0, 2, 16).astype(np.int8)
                    bit_of, partner = core.make_maps(key, "nearest")
                    coded, perm = core.encode_message(message, scheme, key)
                    count = len(coded)
                    positions = select_positions(cover_energy, count, key, strategy)
                    original = cover_codes[0, 0, core.LAYER, torch.from_numpy(positions)].long().numpy()
                    changed = bit_of[original] != coded
                    modified = original.copy()
                    modified[changed] = partner[original[changed]]
                    stego_codes = cover_codes.clone()
                    stego_codes[0, 0, core.LAYER, torch.from_numpy(positions)] = torch.from_numpy(modified)
                    with torch.no_grad():
                        stego_wave = model.decode(stego_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
                        stego_pcm, _ = core._pcm16_roundtrip(stego_wave)
                        stego_re = model.encode(core._signal_tensor(stego_pcm), bandwidth=24.0).audio_codes

                    recv = stego_re[0, 0, core.LAYER, torch.from_numpy(positions)].long().numpy()
                    survivor = float(np.mean(recv == modified))
                    stability = 0.0
                    if strategy == "energy":
                        again = select_positions(frame_energy(model, stego_re), count, key, strategy)
                        stability = float(np.mean(again == positions))

                    results = {}
                    for soft in (False, True):
                        bits_rx = bit_of[recv]
                        if soft:
                            weights = pair_weights(recv, partner, bit_of)
                            deint = np.empty_like(weights)
                            deint[perm] = weights
                            decoded = conv_decode_soft(np.empty_like(bits_rx).__setitem__(slice(None), bits_rx[perm]) or bits_rx[perm], deint, int(scheme[-1]), 16)
                        else:
                            decoded = core.decode_message(bits_rx, scheme, perm, 16)
                        results["soft" if soft else "hard"] = bool(np.array_equal(decoded, message))

                    perceptual = core.perceptual_metrics(cover_pcm, stego_pcm)
                    rows.append({
                        "scheme": scheme, "strategy": strategy, "key": key,
                        "carriers": int(count), "changed": int(changed.sum()),
                        "changed_ratio": round(float(changed.mean()), 3),
                        "survivor": round(survivor, 3),
                        "select_stability": round(stability, 3),
                        "hard": results["hard"], "soft": results["soft"],
                        "seg_snr": round(perceptual["seg_snr_mean_db"], 2),
                        "lsd": round(perceptual["lsd_db"], 2),
                        "nmr": round(perceptual["nmr_median_db"], 1),
                    })
                    print(json.dumps(rows[-1], ensure_ascii=False))

        summary = {}
        for scheme in ("conv13", "conv12"):
            for strategy in ("uniform", "energy"):
                pick = [r for r in rows if r["scheme"] == scheme and r["strategy"] == strategy]
                summary[f"{scheme}|{strategy}"] = {
                    "hard": round(float(np.mean([r["hard"] for r in pick])), 2),
                    "soft": round(float(np.mean([r["soft"] for r in pick])), 2),
                    "changed": round(float(np.mean([r["changed"] for r in pick])), 1),
                    "survivor": round(float(np.mean([r["survivor"] for r in pick])), 3),
                    "seg_snr": round(float(np.mean([r["seg_snr"] for r in pick])), 2),
                    "lsd": round(float(np.mean([r["lsd"] for r in pick])), 2),
                    "stability": round(float(np.mean([r["select_stability"] for r in pick])), 3),
                }
        report[clip.name] = {"rows": rows, "summary": summary}
        print("\n=== summary ===")
        for name, value in summary.items():
            print(name, json.dumps(value, ensure_ascii=False))

    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
