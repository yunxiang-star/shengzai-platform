"""比较 6 / 12 / 24 kbps 下"重建"这一步的听感，并导出三条试听 WAV。

关注点：用户反馈 EnCodec 重建"有人机味"。要回答的是换采样率还是换码率。
本脚本只做一件事：同一条音频，只改 bandwidth（RVQ 层数），其余全同。

输出：
  assets/preview_cover_6k.wav / _12k.wav / _24k.wav
  runtime/bandwidth_quality.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.fft import dct

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import core  # noqa: E402

BANDWIDTHS = [6.0, 12.0, 24.0]
N_MELS = 40


def mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid == lo:
            mid = lo + 1
        if hi == mid:
            hi = mid + 1
        for k in range(lo, min(mid, bank.shape[1])):
            bank[i, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, min(hi, bank.shape[1])):
            bank[i, k] = (hi - k) / max(hi - mid, 1)
    return bank


def mel_cepstrum(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    bank = mel_filterbank(sr, n_fft)
    window = np.hamming(n_fft)
    frames = []
    for start in range(0, len(x) - n_fft, hop):
        spectrum = np.abs(np.fft.rfft(x[start : start + n_fft] * window)) ** 2
        mel = np.log(bank @ spectrum + 1e-10)
        frames.append(dct(mel, type=2, norm="ortho")[:24])
    return np.asarray(frames)


def mcd(reference: np.ndarray, test: np.ndarray, sr: int = core.SR) -> float:
    """梅尔倒谱失真（dB），只统计有声帧（参考能量前 70%）。"""
    a = mel_cepstrum(reference, sr)
    b = mel_cepstrum(test, sr)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    energy = a[:, 0]
    keep = energy >= np.percentile(energy, 30)
    diff = a[keep] - b[keep]
    dist = np.sqrt(np.sum(diff[:, 1:] ** 2, axis=1) * 2)
    return float(np.mean(dist) * 10 / np.log(10) / np.sqrt(2))


def band_limited_lsd(reference: np.ndarray, test: np.ndarray, sr: int = core.SR, hi_hz: float = 8000.0) -> float:
    """只在"有能量"的频点上算对数谱距离，避免近零频点把数字抬到没意义。"""
    n_fft, hop = 512, 160
    window = np.hamming(n_fft)
    cutoff = int(hi_hz / (sr / 2) * (n_fft // 2))
    values = []
    for start in range(0, min(len(reference), len(test)) - n_fft, hop):
        ra = np.abs(np.fft.rfft(reference[start : start + n_fft] * window)) + 1e-9
        rb = np.abs(np.fft.rfft(test[start : start + n_fft] * window)) + 1e-9
        local = ra[:cutoff]
        keep = local >= local.max() * 10 ** (-60 / 20)
        if keep.sum() < 8:
            continue
        values.append(np.mean((20 * np.log10(rb[:cutoff][keep] / local[keep])) ** 2))
    return float(np.sqrt(np.mean(values)))


def main() -> None:
    wav, info = core.read_audio_bytes((HERE / "assets" / "demo.wav").read_bytes())
    model = core.load_model()
    report = {"clip": "assets/demo.wav", "duration_seconds": info["duration_seconds"], "rows": []}
    print(f"clip {info['duration_seconds']:.2f}s  source_sr={info['original_sr']} Hz", flush=True)

    with torch.no_grad():
        for bandwidth in BANDWIDTHS:
            core.BANDWIDTH = bandwidth
            encoded = model.encode(core._signal_tensor(wav), bandwidth=bandwidth)
            shape = tuple(encoded.audio_codes.shape)
            decoded = model.decode(encoded.audio_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
            n = min(len(wav), len(decoded))
            ref, test = wav[:n], decoded[:n]
            perceptual = core.perceptual_metrics(ref, test)
            row = {
                "bandwidth_kbps": bandwidth,
                "codes_shape": list(shape),
                "quantizers": int(shape[2]) if len(shape) == 4 else None,
                "frames": int(shape[-1]),
                "seg_snr_db": round(perceptual["seg_snr_mean_db"], 2),
                "lsd_full_db": round(perceptual["lsd_db"], 2),
                "lsd_to_8k_db": round(band_limited_lsd(ref, test), 2),
                "mcd_db": round(mcd(ref, test), 2),
            }
            path = HERE / "assets" / f"preview_cover_{int(bandwidth)}k.wav"
            core.save_wav(path, decoded)
            row["preview"] = str(path.relative_to(HERE))
            report["rows"].append(row)
            print(
                f"bw={bandwidth:5.1f} codes{shape} Q={row['quantizers']}  "
                f"segSNR={row['seg_snr_db']:5.2f}  LSD(0-8k)={row['lsd_to_8k_db']:5.2f}  MCD={row['mcd_db']:5.2f}  -> {path.name}",
                flush=True,
            )

    core.BANDWIDTH = 6.0
    target = HERE / "runtime" / "bandwidth_quality.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)


if __name__ == "__main__":
    main()
