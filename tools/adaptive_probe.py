"""自适应码率 + 盲码率检测：能不能"能少改就少改"，同时接收端不需要额外信息？

发送端：先试 rate-1/2（44 载波，改动约 22 个）→ 自检通过就交付；否则退 rate-1/3（66 载波）。
接收端：两种码率各解一遍，把解出的 16 bit 重新编码，和读到的比特比对，
        谁的错误率低就是哪一个 —— 不需要任何额外信令。

用法：PYTHONPATH= python tools/adaptive_probe.py [段数] [每段口令数]
输出：runtime/adaptive_probe.json
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


def embed(model, encoded, tokens, bit_map, partner, message, scheme, frames):
    coded, perm = core.encode_message(message, scheme, KEY)
    coded = np.asarray(coded, dtype=np.int8)
    positions = np.linspace(0, frames - 1, len(coded), dtype=np.int64)
    changed = positions[bit_map[tokens[positions]] != coded]
    modified = tokens.copy()
    modified[changed] = partner[tokens[changed]]
    stego = encoded.audio_codes.clone()
    stego[0, 0, core.LAYER] = torch.from_numpy(modified)
    wave = model.decode(stego, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
    pcm, _ = core._pcm16_roundtrip(wave)
    # 发送前自检：走完整重编码信道
    re_tokens = model.encode(core._signal_tensor(pcm), bandwidth=core.BANDWIDTH).audio_codes
    received = bit_map[re_tokens[0, 0, core.LAYER].cpu().numpy().astype(np.int64)[positions]]
    decoded = core.decode_message(received, scheme, perm)
    exact = bool(np.array_equal(decoded, message))
    return {"exact": exact, "changed": int(len(changed)), "pcm": pcm, "scheme": scheme}


def blind_decode(model, pcm, frames, bit_map):
    """两种码率各解一遍，用"重新编码后的比特错误率"挑更可信的那个。"""
    re_tokens = model.encode(core._signal_tensor(pcm), bandwidth=core.BANDWIDTH).audio_codes
    tokens = re_tokens[0, 0, core.LAYER].cpu().numpy().astype(np.int64)
    if len(tokens) != frames:
        frames = len(tokens)
    candidates = []
    for scheme in ("conv12", "conv13"):
        _, perm = core.encode_message(np.zeros(core.PAYLOAD_BITS, dtype=np.int8), scheme, KEY)
        positions = np.linspace(0, frames - 1, core.coded_length(scheme), dtype=np.int64)
        bits = bit_map[tokens[positions]]
        message = core.decode_message(bits, scheme, perm)
        ref, _ = core.encode_message(message, scheme, KEY)
        distance = float(np.mean(bits != np.asarray(ref, dtype=np.int8)))
        candidates.append({"scheme": scheme, "message": message, "error_rate": distance})
    best = min(candidates, key=lambda c: c["error_rate"])
    return best, candidates


def main() -> None:
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    per_clip = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    files = [HERE / "assets" / "demo.wav"] + sorted(SRC.glob("*.wav"))[12 : 12 + n_clips]
    model = core.load_model()
    core.BANDWIDTH = 24.0
    out: dict = {"clips": [], "summary": {}}
    stats = {"ladder_exact": [], "detect_correct": [], "detect_wrong_scheme_but_ok": []}
    chosen_count = {"conv12": 0, "conv13": 0}

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
            rows = []
            for secret in SECRETS[:per_clip]:
                message = core.hex_to_bits(secret)
                pick = None
                for scheme in ("conv12", "conv13"):
                    trial = embed(model, encoded, tokens, bit_map, partner, message, scheme, frames)
                    pick = trial
                    if trial["exact"]:
                        break
                chosen_count[pick["scheme"]] += 1
                best, candidates = blind_decode(model, pick["pcm"], frames, bit_map)
                ok = bool(np.array_equal(best["message"], message))
                rows.append({
                    "secret": secret,
                    "sent_scheme": pick["scheme"],
                    "send_exact": pick["exact"],
                    "detected_scheme": best["scheme"],
                    "detect_ok": ok,
                    "changed": pick["changed"],
                    "margins": {c["scheme"]: round(c["error_rate"], 4) for c in candidates},
                })
                stats["ladder_exact"].append(pick["exact"])
                stats["detect_correct"].append(best["scheme"] == pick["scheme"])
                stats["detect_wrong_scheme_but_ok"].append(
                    best["scheme"] != pick["scheme"] and ok
                )
            out["clips"].append({"source": path.name, "seconds": info["duration_seconds"], "rows": rows})
            print(
                f"{path.name[:18]:18s} {info['duration_seconds']:5.2f}s "
                f"send_exact={np.mean([r['send_exact'] for r in rows])*100:5.1f}% "
                f"detect_ok={np.mean([r['detect_ok'] for r in rows])*100:5.1f}% "
                f"schemes={[r['sent_scheme'] for r in rows]}",
                flush=True,
            )

    out["summary"] = {
        "n": len(stats["ladder_exact"]),
        "send_exact_rate": float(np.mean(stats["ladder_exact"])),
        "blind_decode_ok_rate": float(np.mean(stats["detect_correct"])),
        "scheme_chosen": chosen_count,
    }
    print("\n== 汇总 ==")
    print(f"  发送端自适应后 exact = {out['summary']['send_exact_rate']*100:.1f}%  (n={out['summary']['n']})")
    print(f"  盲检测选对码率         = {out['summary']['blind_decode_ok_rate']*100:.1f}%")
    print(f"  实际选用的码率         = {chosen_count}")

    target = HERE / "runtime" / "adaptive_probe.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)


if __name__ == "__main__":
    main()
