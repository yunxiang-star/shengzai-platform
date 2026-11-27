"""跨多段音频验证 rate-1/2 是否真的优于 rate-1/3。

单段音频上 conv12 既少改又更准，但这可能是"位置运气"。这里换 6 段不同说话人，
每段 8 条口令，两个码率各跑一遍。

用法：PYTHONPATH= python tools/rate_multiclip.py [段数] [每段口令数]
输出：runtime/rate_multiclip.json
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
SECRETS = ["C0DE", "46FF", "1391", "C4DD", "7A3B", "0F2E", "B81D", "55C7"]
SRC = pathlib.Path("D:/语音深伪与隐写/01_检测实验台/samples/real_ms")


def main() -> None:
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    per_clip = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    files = sorted(SRC.glob("*.wav"))[12 : 12 + n_clips]
    model = core.load_model()
    core.BANDWIDTH = 24.0
    out: dict = {"clips": [], "by_scheme": {}}
    agg = {s: [] for s in ("conv12", "conv13")}

    with torch.no_grad():
        for path in files:
            wav, info = core.read_audio_bytes(path.read_bytes())
            encoded = model.encode(core._signal_tensor(wav), bandwidth=core.BANDWIDTH)
            cover_codes = encoded.audio_codes.clone()
            cover_wave = model.decode(cover_codes, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
            cover_pcm, _ = core._pcm16_roundtrip(cover_wave)
            frames = int(cover_codes.shape[3])
            tokens = cover_codes[0, 0, core.LAYER].cpu().numpy().astype(np.int64)
            bit_map, partner = core.make_maps(KEY, "nearest")
            clip_row = {"source": path.name, "seconds": info["duration_seconds"], "frames": frames}

            for scheme in ("conv12", "conv13"):
                rows = []
                for secret in SECRETS[:per_clip]:
                    message = core.hex_to_bits(secret)
                    coded, perm = core.encode_message(message, scheme, KEY)
                    coded = np.asarray(coded, dtype=np.int8)
                    positions = np.linspace(0, frames - 1, len(coded), dtype=np.int64)
                    changed = positions[bit_map[tokens[positions]] != coded]
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
                        "raw": float(np.mean(received == coded)),
                        "exact": bool(np.array_equal(decoded, message)),
                        "seg": round(perc["seg_snr_mean_db"], 2),
                        "lsd": round(perc["lsd_db"], 2),
                    })
                agg[scheme].extend(rows)
                clip_row[scheme] = {
                    "carriers": len(coded),
                    "changed": round(float(np.mean([r["changed"] for r in rows])), 1),
                    "raw": round(float(np.mean([r["raw"] for r in rows])), 3),
                    "exact": float(np.mean([r["exact"] for r in rows])),
                    "seg": round(float(np.mean([r["seg"] for r in rows])), 2),
                    "lsd": round(float(np.mean([r["lsd"] for r in rows])), 2),
                }
            out["clips"].append(clip_row)
            print(
                f"{path.name[:18]:18s} {info['duration_seconds']:5.2f}s "
                f"| conv12 exact={clip_row['conv12']['exact']*100:5.1f}% changed={clip_row['conv12']['changed']:5.1f} seg={clip_row['conv12']['seg']:5.2f} "
                f"| conv13 exact={clip_row['conv13']['exact']*100:5.1f}% changed={clip_row['conv13']['changed']:5.1f} seg={clip_row['conv13']['seg']:5.2f}",
                flush=True,
            )

    print("\n== 汇总 ==")
    for scheme, rows in agg.items():
        summary = {
            "n": len(rows),
            "exact": float(np.mean([r["exact"] for r in rows])),
            "changed": float(np.mean([r["changed"] for r in rows])),
            "raw": float(np.mean([r["raw"] for r in rows])),
            "seg": float(np.mean([r["seg"] for r in rows])),
            "lsd": float(np.mean([r["lsd"] for r in rows])),
        }
        out["by_scheme"][scheme] = summary
        print(
            f"  {scheme}: n={summary['n']} exact={summary['exact']*100:5.1f}% "
            f"changed={summary['changed']:5.1f} seg={summary['seg']:5.2f} lsd={summary['lsd']:4.2f}",
            flush=True,
        )

    target = HERE / "runtime" / "rate_multiclip.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)


if __name__ == "__main__":
    main()
