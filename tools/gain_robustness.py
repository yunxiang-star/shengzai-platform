"""Run a small end-to-end gain-robustness regression on four speech clips.

This checks both sender-side input attenuation and an additional -10 dB gain
change to the saved PCM16 stego file. It does not model lossy codecs or edits.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402

SOURCE_DIR = ROOT / "01_检测实验台" / "samples" / "real_ms"
KEY = "gain-regression"
SECRET = "C0"
INPUT_GAINS = (1.0, 10 ** (-10 / 20))
OUTPUT_GAIN = 10 ** (-10 / 20)


def main() -> None:
    files = [HERE / "assets" / "demo.wav"]
    files += [
        SOURCE_DIR / "real_spk1272_03.wav",
        SOURCE_DIR / "real_spk1673_04.wav",
        SOURCE_DIR / "real_spk1988_05.wav",
    ]
    message = core.hex_to_bits(SECRET, 8)
    rows = []

    for path in files:
        wav, _ = core.read_audio_bytes(path.read_bytes())
        for input_gain in INPUT_GAINS:
            source = np.asarray(wav * input_gain, dtype=np.float32)
            result = core.analyze_embedding(
                source,
                "direct",
                message,
                KEY,
                nonce=f"{path.name}-{input_gain:.6f}",
                payload_bits=8,
            )
            if not result["exact"]:
                result = core.analyze_embedding(
                    source,
                    "hpp",
                    message,
                    KEY,
                    nonce=f"{path.name}-{input_gain:.6f}",
                    payload_bits=8,
                )

            saved, _ = core._pcm16_roundtrip(result["stego_wave"])
            quieter, _ = core._pcm16_roundtrip(saved * OUTPUT_GAIN)
            clean_extract = core.extract_file(
                saved, result["mode"], KEY, payload_bits=8
            )
            gain_extract = core.extract_file(
                quieter, result["mode"], KEY, payload_bits=8
            )
            rows.append(
                {
                    "audio": path.name,
                    "sender_gain_db": round(20 * np.log10(input_gain), 1),
                    "mode": result["mode"],
                    "sender_selfcheck": bool(result["exact"]),
                    "saved_pcm16_exact": clean_extract["hex"] == SECRET,
                    "saved_pcm16_minus10db_exact": gain_extract["hex"] == SECRET,
                    "attempts": int(result["attempts"]),
                    "changed_tokens": int(result["changed_tokens"]),
                    "segment_snr_db": round(
                        result["perceptual"]["seg_snr_mean_db"], 2
                    ),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    report = {
        "protocol": "8-bit direct with HPP fallback; canonical codec input level",
        "n_cases": len(rows),
        "all_sender_selfchecks": all(r["sender_selfcheck"] for r in rows),
        "all_saved_pcm16_exact": all(r["saved_pcm16_exact"] for r in rows),
        "all_minus10db_exact": all(r["saved_pcm16_minus10db_exact"] for r in rows),
        "results": rows,
        "scope": "Four clips; gain changes only. No claim for crop, MP3, or platform transcoding.",
    }
    out = HERE / "runtime" / "gain_robustness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    print(f"saved: {out}")
    if not report["all_sender_selfchecks"] or not report["all_saved_pcm16_exact"] or not report["all_minus10db_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
