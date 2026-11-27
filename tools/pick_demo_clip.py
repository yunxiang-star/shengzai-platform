# -*- coding: utf-8 -*-
"""挑一段「codec 自身失真较小」的示例音频。

判据只看 原始 WAV vs EnCodec 重建（不含任何隐写）：
    seg_snr_mean 越高越好，lsd_db 越低越好，时长 3.5–8 秒。
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

SAMPLES = Path(r"D:/语音深伪与隐写/01_检测实验台/samples/real_ms")


def main():
    model = core.load_model()
    rows = []
    with torch.no_grad():
        for path in sorted(SAMPLES.glob("*.wav")):
            wav, info = core.read_audio_bytes(path.read_bytes(), max_seconds=8.0)
            if not (3.5 <= info["duration_seconds"] <= 8.0):
                continue
            encoded = model.encode(core._signal_tensor(wav), bandwidth=core.BANDWIDTH)
            cover = model.decode(encoded.audio_codes, encoded.audio_scales, None).audio_values
            cover_pcm = cover.squeeze().cpu().numpy()
            n = min(len(wav), len(cover_pcm))
            metrics = pc.describe("codec", wav[:n], cover_pcm[:n])
            rows.append({
                "file": path.name,
                "seconds": info["duration_seconds"],
                "frames": int(encoded.audio_codes.shape[3]),
                "lsd_db": metrics["lsd_db"],
                "seg_snr_mean": metrics["seg_snr"]["mean"],
                "snr_db": metrics["snr_db"],
            })
            print(f"{path.name:24s} {info['duration_seconds']:4.2f}s "
                  f"lsd={metrics['lsd_db']:6.2f} segSNR={metrics['seg_snr']['mean']:6.2f} "
                  f"snr={metrics['snr_db']:6.2f}", flush=True)

    rows.sort(key=lambda r: (-r["seg_snr_mean"], r["lsd_db"]))
    print("\nbest candidates:")
    for row in rows[:6]:
        print(f"  {row['file']:24s} {row['seconds']:4.2f}s segSNR={row['seg_snr_mean']:5.2f} lsd={row['lsd_db']:5.2f}")
    (APP / "runtime" / "clip_quality.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
