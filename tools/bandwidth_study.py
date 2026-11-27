"""带宽（RVQ 层数）对音质与隐写信道的影响。

用户反馈：EnCodec 重建那一段听起来有"人机味"。本脚本回答两件事：
1. 换带宽（不是采样率）能把 codec 底噪降多少；
2. 提高带宽后，隐藏的信道还通不通（direct / HPP 的恢复率与再编码稳定性）。

用法：PYTHONPATH= python tools/bandwidth_study.py [每种配置的消息数]
输出：runtime/bandwidth_study.json + 终端表格
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

import core  # noqa: E402

BANDWIDTHS = [6.0, 12.0, 24.0]
SECRETS = ["C0DE", "46FF", "1391", "C4DD", "7A3B", "0F2E", "B81D", "55C7"]
MODES = ["direct", "hpp"]


def summarize(rows: list[dict], frames: int | None = None, quantizers: int | None = None) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n": len(rows),
        "exact": float(np.mean([r["exact"] for r in rows])),
        "raw_mean": mean("raw"),
        "raw_min": float(np.min([r["raw"] for r in rows])),
        "lsd_embed": mean("lsd_embed"),
        "seg_embed": mean("seg_embed"),
        "lsd_floor": mean("lsd_floor"),
        "seg_floor": mean("seg_floor"),
        "stoi": mean("stoi"),
        "tries": mean("tries"),
        "frames": frames,
        "quantizers": quantizers,
    }


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    wav, info = core.read_audio_bytes((HERE / "assets" / "demo.wav").read_bytes())
    model = core.load_model()
    out = {
        "clip": "assets/demo.wav",
        "duration_seconds": info["duration_seconds"],
        "sample_rate": core.SR,
        "secrets": SECRETS[:repeats],
        "results": {},
    }
    print(f"clip={info['duration_seconds']:.2f}s  sr={core.SR}  repeats={repeats}", flush=True)

    for bandwidth in BANDWIDTHS:
        core.BANDWIDTH = bandwidth
        rows = []
        quantizers = None
        frames = None
        for mode in MODES:
            for i, secret in enumerate(SECRETS[:repeats]):
                t0 = time.time()
                r = core.analyze_embedding(
                    wav, mode, core.hex_to_bits(secret), "demo-2026", nonce=f"bw{bandwidth}{mode}{i}"
                )
                cover = r["cover_wave"]
                stego = r["stego_wave"]
                n = min(len(cover), len(stego))
                embed = core.perceptual_metrics(cover[:n], stego[:n])
                floor = r["perceptual_source"]
                rows.append({
                    "mode": mode,
                    "secret": secret,
                    "exact": bool(r["exact"]),
                    "raw": r["raw_bit_accuracy"],
                    "tries": r.get("attempts", 1),
                    "lsd_embed": embed["lsd_db"],
                    "seg_embed": embed["seg_snr_mean_db"],
                    "audible": embed["audible_fraction"],
                    "lsd_floor": floor["lsd_db"],
                    "seg_floor": floor["seg_snr_mean_db"],
                    "stoi": r["stoi"],
                    "seconds": round(time.time() - t0, 2),
                })
                quantizers = quantizers or r.get("quantizers")
                frames = frames or r.get("frame_count")
            block = [x for x in rows if x["mode"] == mode]
            print(
                f"  bw={bandwidth:5.1f} {mode:6s} exact={sum(x['exact'] for x in block)}/{len(block)} "
                f"raw={np.mean([x['raw'] for x in block]):.3f} "
                f"segEmb={np.mean([x['seg_embed'] for x in block]):6.2f} "
                f"lsdEmb={np.mean([x['lsd_embed'] for x in block]):6.2f} "
                f"segFloor={np.mean([x['seg_floor'] for x in block]):6.2f} "
                f"stoi={np.mean([x['stoi'] for x in block]):.3f}",
                flush=True,
            )
        out["results"][str(bandwidth)] = {
            "bandwidth_kbps": bandwidth,
            "quantizers": quantizers,
            "frames": frames,
            "by_mode": {
                m: summarize([x for x in rows if x["mode"] == m], frames, quantizers) for m in MODES
            },
        }

    core.BANDWIDTH = 6.0
    target = HERE / "runtime" / "bandwidth_study.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)


if __name__ == "__main__":
    main()
