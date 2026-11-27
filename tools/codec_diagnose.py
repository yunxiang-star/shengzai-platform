"""诊断 EnCodec 重建"人机味"到底来自哪里。

分别检查：
1. 重建与原始是否存在时间偏移（若有，任何时域 SNR 都会被严重低估）；
2. 对齐后的分段 SNR / LSD；
3. 各频带能量占比（判断 8–12 kHz 是否只是把上采样噪声编码进去、浪费码率）；
4. 是否削顶。

用法：PYTHONPATH= python tools/codec_diagnose.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.signal import correlate

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import core  # noqa: E402


def best_lag(reference: np.ndarray, test: np.ndarray, max_lag: int = 2000) -> tuple[int, float]:
    """在 ±max_lag 内找使相关最大的偏移（test 相对 reference）。"""
    a = reference.astype(np.float64)
    b = test.astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    # 用 FFT 相关，长信号也不慢
    corr = correlate(b, a, mode="full", method="fft")
    lags = np.arange(-len(a) + 1, len(b))
    mask = np.abs(lags) <= max_lag
    idx = np.argmax(corr[mask])
    lag = int(lags[mask][idx])
    # 归一化相关系数
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-12
    return lag, float(corr[mask][idx] / denom)


def band_energy(x: np.ndarray, sr: int = core.SR) -> dict:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    total = spectrum.sum() + 1e-20
    out = {}
    for lo, hi in [(0, 4000), (4000, 8000), (8000, 12000), (12000, 12000.5)]:
        sel = (freqs >= lo) & (freqs < hi)
        out[f"{lo // 1000}-{hi // 1000 if hi < 12000 else 12}k"] = float(spectrum[sel].sum() / total)
    out["centroid_hz"] = float((spectrum * freqs).sum() / total)
    return out


def main() -> None:
    wav, info = core.read_audio_bytes((HERE / "assets" / "demo.wav").read_bytes())
    model = core.load_model()
    report = {
        "clip": "assets/demo.wav",
        "source_sr_hz": int(info.get("original_sr", core.SR)),
        "duration_seconds": info["duration_seconds"],
        "source_band": band_energy(wav),
        "by_bandwidth": {},
    }
    print(f"clip {info['duration_seconds']:.2f}s  peak={np.abs(wav).max():.3f}", flush=True)
    print("source band energy:", {k: round(v, 4) for k, v in report["source_band"].items()}, flush=True)

    with torch.no_grad():
        for bandwidth in (6.0, 12.0, 24.0):
            core.BANDWIDTH = bandwidth
            encoded = model.encode(core._signal_tensor(wav), bandwidth=bandwidth)
            decoded = model.decode(encoded.audio_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
            n = min(len(wav), len(decoded))
            ref, test = wav[:n], decoded[:n]
            lag, corr_peak = best_lag(ref, test)
            if lag > 0:
                ref_a, test_a = ref[lag:], test[: n - lag]
            elif lag < 0:
                ref_a, test_a = ref[: n + lag], test[-lag:]
            else:
                ref_a, test_a = ref, test
            m = min(len(ref_a), len(test_a))
            ref_a, test_a = ref_a[:m], test_a[:m]
            metrics_raw = core.perceptual_metrics(ref, test)
            metrics_aligned = core.perceptual_metrics(ref_a, test_a)
            entry = {
                "bandwidth_kbps": bandwidth,
                "quantizers": int(encoded.audio_codes.shape[1]),
                "frames": int(encoded.audio_codes.shape[3]),
                "lag_samples": lag,
                "lag_ms": round(lag / core.SR * 1000, 2),
                "corr_peak": round(corr_peak, 4),
                "seg_snr_raw_db": round(metrics_raw["seg_snr_mean_db"], 2),
                "seg_snr_aligned_db": round(metrics_aligned["seg_snr_mean_db"], 2),
                "lsd_raw_db": round(metrics_raw["lsd_db"], 2),
                "lsd_aligned_db": round(metrics_aligned["lsd_db"], 2),
                "decoded_peak": round(float(np.abs(decoded).max()), 4),
                "clip_fraction": float(np.mean(np.abs(decoded) >= 0.999)),
                "band": band_energy(decoded),
            }
            report["by_bandwidth"][str(bandwidth)] = entry
            print(
                f"bw={bandwidth:5.1f} Q={entry['quantizers']:2d} lag={lag:+4d} ({entry['lag_ms']:+.2f} ms) "
                f"corr={corr_peak:.3f} segSNR {entry['seg_snr_raw_db']:5.2f} -> {entry['seg_snr_aligned_db']:5.2f} dB  "
                f"LSD {entry['lsd_raw_db']:5.2f} -> {entry['lsd_aligned_db']:5.2f} dB  "
                f"peak={entry['decoded_peak']:.3f} centroid={entry['band']['centroid_hz']:.0f} Hz",
                flush=True,
            )
            print("      band:", {k: round(v, 4) for k, v in entry["band"].items()}, flush=True)

    core.BANDWIDTH = 6.0
    target = HERE / "runtime" / "codec_diagnose.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)


if __name__ == "__main__":
    main()
