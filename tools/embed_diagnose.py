"""诊断：改动代价到底在哪里。

三个问题：
1. 帧能量与"改动后是否可听"的关系——安静帧真的更便宜吗？
2. 帧能量与"重编码后 token 是否存活"的关系——安静帧更稳还是更脆？
3. 码本配对距离：全局最小权匹配 vs 哈希标签 + 最近异号码字，哪个更近？

用法：PYTHONPATH= python tools/embed_diagnose.py
输出：runtime/embed_diagnose.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import core  # noqa: E402

SECRET = "C0DE"
KEY = "demo-2026"


def frame_energy(wave: np.ndarray, frames: int) -> np.ndarray:
    """把波形按 token 帧（hop=320）切块，返回每帧 RMS 能量。"""
    hop = 320
    usable = min(len(wave) // hop, frames)
    blocks = wave[: usable * hop].reshape(usable, hop).astype(np.float64)
    energy = np.sqrt((blocks**2).mean(axis=1))
    if usable < frames:
        energy = np.pad(energy, (0, frames - usable), constant_values=np.nan)
    return energy


def decile_table(values: np.ndarray, metric: np.ndarray, label: str) -> list[dict]:
    ok = np.isfinite(values) & np.isfinite(metric)
    values, metric = values[ok], metric[ok]
    order = np.argsort(values)
    rows = []
    for q in range(10):
        lo = int(len(order) * q / 10)
        hi = int(len(order) * (q + 1) / 10)
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        rows.append({
            "decile": q + 1,
            "value_range": [float(values[idx].min()), float(values[idx].max())],
            "mean": float(metric[idx].mean()),
        })
    print(f"\n{label}")
    for row in rows:
        print(f"  D{row['decile']:<2} 值域 [{row['value_range'][0]:.4f}, {row['value_range'][1]:.4f}] → {row['mean']:.4f}")
    return rows


def main() -> None:
    wav, info = core.read_audio_bytes((HERE / "assets" / "demo.wav").read_bytes())
    model = core.load_model()
    report: dict = {"clip": "assets/demo.wav", "duration_seconds": info["duration_seconds"]}

    with torch.no_grad():
        core.BANDWIDTH = 24.0
        encoded = model.encode(core._signal_tensor(wav), bandwidth=24.0)
        cover_codes = encoded.audio_codes.clone()
        cover_wave = model.decode(cover_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
        cover_pcm, _ = core._pcm16_roundtrip(cover_wave)
        frames = int(cover_codes.shape[3])
        layer = core.LAYER
        cover_tokens = cover_codes[0, 0, layer].cpu().numpy().astype(np.int64)
        print(f"frames={frames}  codebook=1024", flush=True)

        energy = frame_energy(cover_pcm, frames)
        report["energy_stats"] = {
            "min": float(np.nanmin(energy)),
            "median": float(np.nanmedian(energy)),
            "max": float(np.nanmax(energy)),
            "silence_fraction_below_1e-3": float(np.nanmean(energy < 1e-3)),
        }

        # ---- 1) 统一改所有 token，看哪一帧的改动代价最大 ----
        bit_of, partner = core.make_maps(KEY, core.PAIRING_FOR_MODE["direct"])
        flipped = partner[cover_tokens].astype(np.int64)
        all_flipped = cover_codes.clone()
        all_flipped[0, 0, layer] = torch.from_numpy(flipped)
        flipped_wave = model.decode(all_flipped, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
        hop = 320
        n = min(len(cover_pcm), len(flipped_wave))
        diff = cover_pcm[:n] - flipped_wave[:n]
        usable = min(n // hop, frames)
        err = np.sqrt((diff[: usable * hop].reshape(usable, hop).astype(np.float64) ** 2).mean(axis=1))
        err = np.pad(err, (0, frames - usable), constant_values=np.nan)
        # 相对代价：该帧改动误差 / 该帧信号能量
        relative = err / (energy * 0.0 + np.maximum(energy, 1e-6))
        report["error_vs_energy"] = decile_table(energy, err, "改动误差（绝对）按帧能量分十分位：")
        report["relative_vs_energy"] = decile_table(energy, relative, "改动误差 / 帧能量（相对量）按能量分十分位：")

        # ---- 2) 重编码后 token 存活率 vs 能量 ----
        reencoded = model.encode(core._signal_tensor(cover_pcm), bandwidth=24.0).audio_codes
        re_tokens = reencoded[0, 0, layer].cpu().numpy().astype(np.int64)
        m = min(len(re_tokens), len(cover_tokens))
        survival = (re_tokens[:m] == cover_tokens[:m]).astype(np.float64)
        report["survival_overall"] = float(survival.mean())
        report["survival_vs_energy"] = decile_table(
            energy[:m], survival, f"重编码后 token 存活率（总体 {survival.mean():.3f}）按帧能量分十分位："
        )

        # ---- 3) 码本距离：匹配 vs 哈希 ----
        codebook = model.quantizer.layers[layer].codebook.embed.squeeze().float().cpu().numpy()
        matched = np.linalg.norm(codebook - codebook[partner], axis=1)
        # 哈希标签：每个码字独立 0/1，取最近的异号码字
        rng = np.random.default_rng(core.key_seed(KEY, "hash-label"))
        labels = rng.integers(0, 2, size=1024)
        nearest_opposite = np.empty(1024)
        for c in range(1024):
            candidates = np.where(labels != labels[c])[0]
            d = ((codebook[candidates] - codebook[c]) ** 2).sum(axis=1)
            nearest_opposite[c] = np.sqrt(d.min())
        report["codebook_distance"] = {
            "matched_mean": float(matched.mean()),
            "matched_median": float(np.median(matched)),
            "hash_nearest_mean": float(nearest_opposite.mean()),
            "hash_nearest_median": float(np.median(nearest_opposite)),
            "hash_label_balance": [int((labels == 0).sum()), int((labels == 1).sum())],
        }
        print("\n码本距离（128 维欧氏）：")
        print(f"  最小权匹配      mean={matched.mean():.4f} median={np.median(matched):.4f}")
        print(f"  哈希+最近异号   mean={nearest_opposite.mean():.4f} median={np.median(nearest_opposite):.4f}")

    core.BANDWIDTH = 24.0
    target = HERE / "runtime" / "embed_diagnose.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)


if __name__ == "__main__":
    main()
