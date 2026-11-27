"""STC 最小失真嵌入 vs 现有直接替换：同一个信道、同一条音频。

要回答的问题：把 UNIWARD 那套「代价函数 + STC」搬到 token 域，能不能既少改、
又照样从"解码→PCM16→再编码"之后读回来。

用法：PYTHONPATH= python tools/stc_probe.py [口令条数]
输出：runtime/stc_probe.json
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
import stc  # noqa: E402

KEY = "demo-2026"
SECRETS = ["C0DE", "46FF", "1391", "C4DD", "7A3B", "0F2E"]
HOP = 320


def frame_energy(wave: np.ndarray, frames: int) -> np.ndarray:
    usable = min(len(wave) // HOP, frames)
    blocks = wave[: usable * HOP].reshape(usable, HOP).astype(np.float64)
    energy = np.sqrt((blocks**2).mean(axis=1)) + 1e-6
    return np.pad(energy, (0, max(frames - usable, 0)), constant_values=float(energy.mean()))


def hash_maps(key: str, codebook: np.ndarray):
    """哈希标签 + 每个码字最近的异号码字（不要求一一配对）。"""
    rng = np.random.default_rng(core.key_seed(key, "hash-label"))
    labels = rng.integers(0, 2, size=1024).astype(np.int8)
    nearest = np.empty(1024, dtype=np.int64)
    for c in range(1024):
        candidates = np.where(labels != labels[c])[0]
        d = ((codebook[candidates] - codebook[c]) ** 2).sum(axis=1)
        nearest[c] = int(candidates[int(d.argmin())])
    return labels, nearest


def bits_to_targets(bits: np.ndarray, h: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    nb = len(bits) // h
    packed = np.packbits(bits[: nb * h].reshape(nb, h), axis=1, bitorder="little")
    return packed[:, 0].astype(np.int64)


def targets_to_bits(targets: np.ndarray, h: int) -> np.ndarray:
    raw = np.unpackbits(np.asarray(targets, dtype=np.uint8)[:, None], axis=1, bitorder="little")
    return raw[:, :h].reshape(-1).astype(np.int8)


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    wav, info = core.read_audio_bytes((HERE / "assets" / "demo.wav").read_bytes())
    model = core.load_model()
    core.BANDWIDTH = 24.0
    out: dict = {"clip": "assets/demo.wav", "bandwidth_kbps": 24.0, "results": {}}

    with torch.no_grad():
        encoded = model.encode(core._signal_tensor(wav), bandwidth=core.BANDWIDTH)
        cover_codes = encoded.audio_codes.clone()
        cover_wave = model.decode(cover_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
        cover_pcm, _ = core._pcm16_roundtrip(cover_wave)
        frames = int(cover_codes.shape[3])
        tokens = cover_codes[0, 0, core.LAYER].cpu().numpy().astype(np.int64)
        energy = frame_energy(cover_pcm, frames)
        print(f"frames={frames}  codebook=1024", flush=True)

        codebook = model.quantizer.layers[core.LAYER].codebook.embed.squeeze().float().cpu().numpy()
        matched_bits, matched_partner = core.make_maps(KEY, "nearest")
        hash_bits, hash_partner = hash_maps(KEY, codebook)

        def measure(stego_codes):
            wave = model.decode(stego_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
            pcm, _ = core._pcm16_roundtrip(wave)
            m = min(len(cover_pcm), len(pcm))
            return pcm, core.perceptual_metrics(cover_pcm[:m], pcm[:m])

        def reencode_tokens(pcm):
            codes = model.encode(core._signal_tensor(pcm), bandwidth=core.BANDWIDTH).audio_codes
            return codes[0, 0, core.LAYER].cpu().numpy().astype(np.int64)

        for secret in SECRETS[:repeats]:
            message = core.hex_to_bits(secret)
            coded, perm = core.encode_message(message, core.DIRECT_SCHEME, KEY)
            coded = np.asarray(coded, dtype=np.int8)
            print(f"\n== {secret}  coded={len(coded)} bit", flush=True)

            for name, bit_map, partner in (
                ("direct-matched", matched_bits, matched_partner),
                ("direct-hash", hash_bits, hash_partner),
            ):
                positions = np.linspace(0, frames - 1, len(coded), dtype=np.int64)
                cover_bits = bit_map[tokens[positions]]
                changed = positions[cover_bits != coded]
                modified = tokens.copy()
                modified[changed] = partner[tokens[changed]]
                stego = cover_codes.clone()
                stego[0, 0, core.LAYER] = torch.from_numpy(modified)
                pcm, perc = measure(stego)
                received = bit_map[reencode_tokens(pcm)[positions]]
                decoded = core.decode_message(received, core.DIRECT_SCHEME, perm)
                row = {
                    "secret": secret,
                    "changed": int(len(changed)),
                    "raw_accuracy": float(np.mean(received == coded)),
                    "exact": bool(np.array_equal(decoded, message)),
                    "seg_snr_db": round(perc["seg_snr_mean_db"], 2),
                    "lsd_db": round(perc["lsd_db"], 2),
                }
                out["results"].setdefault(name, []).append(row)
                print(f"  {name:18s} changed={row['changed']:3d} raw={row['raw_accuracy']:.3f} exact={row['exact']} "
                      f"segSNR={row['seg_snr_db']:5.2f} lsd={row['lsd_db']:4.2f}", flush=True)

            for name, n_use, h, weight in (
                ("stc-all-h6", frames, 6, False),
                ("stc-all-h6-energy", frames, 6, True),
                ("stc-165-h6", 165, 6, False),
            ):
                nb = len(coded) // h
                w = n_use // nb
                n = nb * w
                if w < 2:
                    print(f"  {name:18s} skipped (w={w})", flush=True)
                    continue
                positions = np.linspace(0, frames - 1, n, dtype=np.int64)
                cover_bits = matched_bits[tokens[positions]].astype(np.int64)
                columns = stc.build_columns(core.key_seed(KEY, f"stc|{name}|{frames}"), h, w, nb)
                targets = bits_to_targets(coded, h)
                delta = targets ^ stc.syndrome_of(cover_bits, columns)
                rho = 1.0 / energy[positions] if weight else np.ones(n)
                e, _cost = stc.embed(cover_bits, rho, delta, columns, h)
                flip = positions[e]
                modified = tokens.copy()
                modified[flip] = matched_partner[tokens[flip]]
                stego = cover_codes.clone()
                stego[0, 0, core.LAYER] = torch.from_numpy(modified)
                pcm, perc = measure(stego)
                syn = stc.syndrome_of(matched_bits[reencode_tokens(pcm)[positions]].astype(np.int64), columns)
                decoded = core.decode_message(
                    targets_to_bits(syn, h), core.DIRECT_SCHEME, perm
                )
                row = {
                    "secret": secret,
                    "changed": int(len(flip)),
                    "syndrome_accuracy": float(np.mean(syn == targets)),
                    "exact": bool(np.array_equal(decoded, message)),
                    "seg_snr_db": round(perc["seg_snr_mean_db"], 2),
                    "lsd_db": round(perc["lsd_db"], 2),
                }
                out["results"].setdefault(name, []).append(row)
                print(f"  {name:18s} changed={row['changed']:3d} syn={row['syndrome_accuracy']:.3f} exact={row['exact']} "
                      f"segSNR={row['seg_snr_db']:5.2f} lsd={row['lsd_db']:4.2f}", flush=True)

    target = HERE / "runtime" / "stc_probe.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)

    print("\n== 汇总 ==")
    for name, rows in out["results"].items():
        changed = np.mean([r["changed"] for r in rows])
        exact = np.mean([r["exact"] for r in rows])
        seg = np.mean([r["seg_snr_db"] for r in rows])
        lsd = np.mean([r["lsd_db"] for r in rows])
        print(f"  {name:20s} changed={changed:5.1f}  exact={exact*100:5.1f}%  segSNR={seg:5.2f}  lsd={lsd:4.2f}")


if __name__ == "__main__":
    main()
