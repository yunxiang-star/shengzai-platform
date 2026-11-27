# -*- coding: utf-8 -*-
"""感知失真诊断：把“人耳能听出来”拆成可测的量。

三组对照：
  A. 原始 WAV  vs  EnCodec 重建（codec 自身的失真，属于底座噪声）
  B. 重建公开版 vs 藏字版（嵌入引入的失真，我们真正要负责的部分）
  C. 逐帧噪声与掩蔽阈值的比值（NMR 粗估，判断是否可能被听到）

指标：LSD（对数谱距离，dB）、分段 SNR、噪声-掩蔽比、被改 token 的码本距离。
PESQ 在本机装不上（无 cp313 wheel，且缺 MSVC io.h），所以不伪造该指标。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import core  # noqa: E402

import numpy as np  # noqa: E402

SR = core.SR
N_FFT = 1024
HOP = 256

# 近似 Bark 带边界（Hz），覆盖 0–12 kHz
BARK_EDGES = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270,
                       1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300,
                       6400, 7700, 9500, 12000])


def stft_mag(x: np.ndarray) -> np.ndarray:
    window = np.hanning(N_FFT)
    frames = 1 + max(0, (len(x) - N_FFT) // HOP)
    out = np.empty((frames, N_FFT // 2 + 1), dtype=np.float64)
    for i in range(frames):
        segment = x[i * HOP: i * HOP + N_FFT] * window
        out[i] = np.abs(np.fft.rfft(segment))
    return out


def lsd_db(ref: np.ndarray, test: np.ndarray) -> float:
    a = np.maximum(stft_mag(ref), 1e-10)
    b = np.maximum(stft_mag(test), 1e-10)
    frames = min(len(a), len(b))
    diff = 20 * np.log10(a[:frames]) - 20 * np.log10(b[:frames])
    return float(np.mean(np.sqrt(np.mean(diff ** 2, axis=1))))


def seg_snr_db(ref: np.ndarray, test: np.ndarray, frame: int = 480, hop: int = 240):
    n = min(len(ref), len(test))
    values = []
    for start in range(0, n - frame, hop):
        r = ref[start:start + frame]
        e = test[start:start + frame]
        noise = r - e
        pn = float(np.mean(noise ** 2))
        ps = float(np.mean(r ** 2))
        if ps < 1e-8:
            continue
        snr = 10 * np.log10(ps / max(pn, 1e-12))
        values.append(float(np.clip(snr, -10, 35)))
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "min": float(values.min()),
    }


def band_energies(power: np.ndarray) -> np.ndarray:
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    bands = []
    for lo, hi in zip(BARK_EDGES[:-1], BARK_EDGES[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            bands.append(np.zeros(power.shape[0]))
            continue
        bands.append(power[:, mask].sum(axis=1))
    return np.stack(bands, axis=1)


def nmr(ref: np.ndarray, test: np.ndarray, offsets=(-18.0, -24.0, -30.0)):
    """噪声-掩蔽比粗估：掩蔽阈值取同一时频块内公开版能量下移 X dB。"""
    a = stft_mag(ref) ** 2
    noise = (stft_mag(ref) - stft_mag(test)) ** 2  # 幅度谱差，保守近似
    frames = min(len(a), len(noise))
    eb = band_energies(a[:frames])
    nb = band_energies(noise[:frames])
    out = {}
    for offset in offsets:
        threshold = eb * (10 ** (offset / 10.0))
        # 绝对听阈近似：每带 ~ -80 dBFS 量级，避免静音段被无限放大
        threshold = np.maximum(threshold, 1e-10)
        ratio = 10 * np.log10(np.maximum(nb, 1e-16) / threshold)
        audible = float(np.mean(ratio > 0))
        out[f"offset_{abs(int(offset))}db"] = {
            "audible_fraction": audible,
            "nmr_median_db": float(np.median(ratio)),
            "nmr_p95_db": float(np.percentile(ratio, 95)),
            "nmr_max_db": float(np.max(ratio)),
        }
    return out


def token_distance_stats(model, changed_tokens: np.ndarray):
    codebook = model.quantizer.layers[core.LAYER].codebook.embed.squeeze().float().cpu()
    if len(changed_tokens) == 0:
        return {}
    vectors = codebook[changed_tokens]
    norms = vectors.norm(dim=1).numpy()
    return {
        "count": int(len(changed_tokens)),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }


def describe(name, ref, test):
    n = min(len(ref), len(test))
    ref, test = np.asarray(ref[:n], dtype=np.float64), np.asarray(test[:n], dtype=np.float64)
    return {
        "pair": name,
        "lsd_db": lsd_db(ref, test),
        "seg_snr": seg_snr_db(ref, test),
        "snr_db": float(10 * np.log10(np.mean(ref ** 2) / max(np.mean((ref - test) ** 2), 1e-16))),
        "max_abs_diff": float(np.max(np.abs(ref - test))),
        "nmr": nmr(ref, test),
    }


def main():
    wav, info = core.read_audio_bytes((APP / "assets" / "demo.wav").read_bytes())
    model = core.load_model()
    message = core.hex_to_bits("C0DE")

    results = {"clip": "demo.wav", "seconds": info["duration_seconds"], "pairs": []}
    with core.MODEL_LOCK:
        import torch
        with torch.no_grad():
            encoded = model.encode(core._signal_tensor(wav), bandwidth=core.BANDWIDTH)
            cover = model.decode(encoded.audio_codes, encoded.audio_scales, None).audio_values
            cover_pcm = cover.squeeze().cpu().numpy()

    results["pairs"].append(describe("source vs EnCodec重建（codec 自身）", wav[:len(cover_pcm)], cover_pcm))

    for mode in ("direct", "hpp"):
        result = core.analyze_embedding(wav, mode, message, "demo-2026", nonce="perceptual")
        ref = result["cover_wave"]
        test = result["stego_wave"]
        entry = describe(f"重建公开 vs 藏字（{mode}）", ref, test)
        entry["mode"] = mode
        entry["stoi"] = float(result["stoi"])
        entry["changed_tokens"] = int(result["changed_tokens"])
        entry["carrier_positions"] = int(result["carrier_positions"])
        entry["hist_l1_sender"] = float(result["sender_hist_l1"])
        entry["recovered"] = result["recovered_hex"]
        entry["exact"] = bool(result["exact"])
        # 被改 token 与原始 token 的码本距离
        preview = result["token_preview"]
        results.setdefault("changed_preview", {})[mode] = [
            {"frame": p["frame"], "cover": p["cover"], "stego": p["stego"]} for p in preview[:8]
        ]
        results["pairs"].append(entry)
        print(f"{mode}: exact={result['exact']} stoi={result['stoi']:.3f} "
              f"lsd={entry['lsd_db']:.2f}dB segSNR={entry['seg_snr']['mean']:.2f}dB "
              f"snr={entry['snr_db']:.2f}dB changed={entry['changed_tokens']}/{entry['carrier_positions']}", flush=True)

    out = APP / "runtime" / "perceptual.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(results["pairs"], ensure_ascii=False, indent=2)[:2600])
    print("saved", out)


if __name__ == "__main__":
    main()
