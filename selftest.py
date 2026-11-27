# -*- coding: utf-8 -*-
"""不开网页也能验证：示例音频真的能藏、能从 PCM16 回读、能再提取。"""
import json
from pathlib import Path

import core

HERE = Path(__file__).resolve().parent
SECRET = "C0"
KEY = "demo-2026"


def main():
    wav, info = core.read_audio_bytes((HERE / "assets" / "demo.wav").read_bytes())
    message = core.hex_to_bits(SECRET)
    out_dir = HERE / "runtime" / "_selftest"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"audio": info, "secret": SECRET, "key_length": len(KEY), "modes": {}}
    failed = False
    for mode in ["direct", "hpp"]:
        result = core.analyze_embedding(wav, mode, message, KEY)
        core.save_wav(out_dir / f"{mode}_cover.wav", result["cover_wave"])
        stego_path = out_dir / f"{mode}_stego.wav"
        core.save_wav(stego_path, result["stego_wave"])
        saved, _ = core.read_audio_bytes(stego_path.read_bytes())
        extracted = core.extract_file(saved, mode, KEY, payload_bits=len(message))
        quieter, _ = core._pcm16_roundtrip(saved * (10 ** (-10 / 20)))
        gain_extracted = core.extract_file(quieter, mode, KEY, payload_bits=len(message))
        item = {
            "scheme": result["scheme"],
            "exact_after_roundtrip": result["exact"],
            "recovered_hex": result["recovered_hex"],
            "raw_bit_accuracy": result["raw_bit_accuracy"],
            "stoi": result["stoi"],
            "sender_hist_l1": result["sender_hist_l1"],
            "receiver_hist_l1": result["receiver_hist_l1"],
            "changed_tokens": result["changed_tokens"],
            "carrier_positions": result["carrier_positions"],
            "extract_from_saved_wav": extracted["hex"],
            "extract_exact": extracted["hex"] == SECRET,
            "extract_after_minus_10db": gain_extracted["hex"],
            "gain_adjustment_exact": gain_extracted["hex"] == SECRET,
        }
        if mode == "hpp" and result["sender_hist_l1"] > 1e-12:
            item["ERROR"] = "HPP 发送端直方图不再严格保持"
            failed = True
        if not item["exact_after_roundtrip"] or not item["extract_exact"] or not item["gain_adjustment_exact"]:
            item["ERROR"] = "消息没有完整恢复"
            failed = True
        report["modes"][mode] = item
    report["ok"] = not failed
    (out_dir / "selftest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
