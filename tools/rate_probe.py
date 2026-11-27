"""在"能解出来"的前提下把改动压到最少：比较不同码率与配对方式。

STC 把改动从 32 降到 17，但 syndrome 被信道擦除打烂（见 tools/stc_probe.py）。
既然不能靠扩散，就只能靠"少带冗余"：码率越低，需要的载波越少、改动越少。
这里量一下 rate-1/2（44 位）到底还能不能从重编码后读回。

用法：PYTHONPATH= python tools/rate_probe.py [口令条数]
输出：runtime/rate_probe.json
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

KEY = "demo-2026"
SECRETS = ["C0DE", "46FF", "1391", "C4DD", "7A3B", "0F2E", "B81D", "55C7", "9E40", "3210"]


def hash_maps(key: str, codebook: np.ndarray):
    rng = np.random.default_rng(core.key_seed(key, "hash-label"))
    labels = rng.integers(0, 2, size=1024).astype(np.int8)
    nearest = np.empty(1024, dtype=np.int64)
    for c in range(1024):
        candidates = np.where(labels != labels[c])[0]
        d = ((codebook[candidates] - codebook[c]) ** 2).sum(axis=1)
        nearest[c] = int(candidates[int(d.argmin())])
    return labels, nearest


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    wav, info = core.read_audio_bytes((HERE / "assets" / "demo.wav").read_bytes())
    model = core.load_model()
    core.BANDWIDTH = 24.0
    out: dict = {"clip": "assets/demo.wav", "results": {}}

    with torch.no_grad():
        encoded = model.encode(core._signal_tensor(wav), bandwidth=core.BANDWIDTH)
        cover_codes = encoded.audio_codes.clone()
        cover_wave = model.decode(cover_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
        cover_pcm, _ = core._pcm16_roundtrip(cover_wave)
        frames = int(cover_codes.shape[3])
        tokens = cover_codes[0, 0, core.LAYER].cpu().numpy().astype(np.int64)
        codebook = model.quantizer.layers[core.LAYER].codebook.embed.squeeze().float().cpu().numpy()
        matched_bits, matched_partner = core.make_maps(KEY, "nearest")
        hash_bits, hash_partner = hash_maps(KEY, codebook)
        print(f"frames={frames}", flush=True)

        for scheme in ("conv12", "conv13"):
            for map_name, bit_map, partner in (
                ("matched", matched_bits, matched_partner),
                ("hash", hash_bits, hash_partner),
            ):
                name = f"{scheme}-{map_name}"
                rows = []
                for secret in SECRETS[:repeats]:
                    message = core.hex_to_bits(secret)
                    coded, perm = core.encode_message(message, scheme, KEY)
                    coded = np.asarray(coded, dtype=np.int8)
                    positions = np.linspace(0, frames - 1, len(coded), dtype=np.int64)
                    cover_bits = bit_map[tokens[positions]]
                    changed = positions[cover_bits != coded]
                    modified = tokens.copy()
                    modified[changed] = partner[tokens[changed]]
                    stego = cover_codes.clone()
                    stego[0, 0, core.LAYER] = torch.from_numpy(modified)
                    wave = model.decode(stego, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
                    pcm, _ = core._pcm16_roundtrip(wave)
                    re_tokens = model.encode(core._signal_tensor(pcm), bandwidth=core.BANDWIDTH).audio_codes
                    received = bit_map[re_tokens[0, 0, core.LAYER].cpu().numpy().astype(np.int64)[positions]]
                    decoded = core.decode_message(received, scheme, perm)
                    m = min(len(cover_pcm), len(pcm))
                    perc = core.perceptual_metrics(cover_pcm[:m], pcm[:m])
                    rows.append({
                        "secret": secret,
                        "changed": int(len(changed)),
                        "raw_accuracy": float(np.mean(received == coded)),
                        "exact": bool(np.array_equal(decoded, message)),
                        "seg_snr_db": round(perc["seg_snr_mean_db"], 2),
                        "lsd_db": round(perc["lsd_db"], 2),
                    })
                out["results"][name] = rows
                changed = np.mean([r["changed"] for r in rows])
                exact = np.mean([r["exact"] for r in rows])
                seg = np.mean([r["seg_snr_db"] for r in rows])
                lsd = np.mean([r["lsd_db"] for r in rows])
                raw = np.mean([r["raw_accuracy"] for r in rows])
                print(
                    f"  {name:14s} carriers={len(coded):3d} changed={changed:5.1f} raw={raw:.3f} "
                    f"exact={exact*100:5.1f}% segSNR={seg:5.2f} lsd={lsd:4.2f}",
                    flush=True,
                )

    target = HERE / "runtime" / "rate_probe.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)


if __name__ == "__main__":
    main()
