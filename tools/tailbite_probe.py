"""咬尾卷积码（tail-biting）：把 6 个已知零尾比特省掉。

现状：16 bit 消息 + 6 个零尾 = 22 步 × 3 = 66 个载波；那 18 个尾比特的载波
本质上在传"已知的 0"，纯属浪费。
咬尾把初始状态设成消息末 6 位，于是不需要尾巴：16 × 3 = 48 个载波。

代价：译码要试 64 个初始状态（复杂度 ×64，仍然很便宜）。
本脚本验证：48 载波是否既少改、又不掉恢复率。

用法：PYTHONPATH= python tools/tailbite_probe.py [段数] [每段口令数]
输出：runtime/tailbite_probe.json
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
INF = 10**9


def tb_encode(message: np.ndarray, rate: int) -> np.ndarray:
    generators = core.conv_generators(rate)
    message = np.asarray(message, dtype=np.int8)
    state = 0
    for bit in message[-6:]:
        state = ((state << 1) | int(bit)) & 0x3F
    output = []
    for bit in message:
        register = (state << 1) | int(bit)
        output.extend(core.parity(register & g) for g in generators)
        state = register & 0x3F
    return np.asarray(output, dtype=np.int8)


def tb_decode(received: np.ndarray, rate: int, payload_bits: int) -> np.ndarray:
    generators = core.conv_generators(rate)
    steps = payload_bits
    best_cost = INF
    best = None
    for start in range(64):
        costs = np.full(64, INF, dtype=np.int64)
        costs[start] = 0
        prev_state = np.zeros((steps, 64), dtype=np.int16)
        prev_bit = np.zeros((steps, 64), dtype=np.int8)
        for step in range(steps):
            new_costs = np.full(64, INF, dtype=np.int64)
            observed = received[step * rate : (step + 1) * rate]
            for state in range(64):
                if costs[state] >= INF:
                    continue
                for bit in (0, 1):
                    register = (state << 1) | bit
                    expected = [core.parity(register & g) for g in generators]
                    next_state = register & 0x3F
                    cost = int(costs[state]) + sum(int(a != b) for a, b in zip(expected, observed))
                    if cost < new_costs[next_state]:
                        new_costs[next_state] = cost
                        prev_state[step, next_state] = state
                        prev_bit[step, next_state] = bit
            costs = new_costs
        if costs[start] < best_cost:
            best_cost = int(costs[start])
            best = (start, prev_state.copy(), prev_bit.copy())
    if best is None:
        return np.zeros(payload_bits, dtype=np.int8)
    state, prev_state, prev_bit = best
    decoded = np.zeros(steps, dtype=np.int8)
    for step in range(steps - 1, -1, -1):
        decoded[step] = prev_bit[step, state]
        state = prev_state[step, state]
    return decoded


def tb_perm(key: str) -> np.ndarray:
    rng = np.random.default_rng(core.key_seed(key, "interleave"))
    return rng.permutation((core.PAYLOAD_BITS) * 3)


def main() -> None:
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    per_clip = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    files = [HERE / "assets" / "demo.wav"] + sorted(SRC.glob("*.wav"))[12 : 12 + n_clips]
    model = core.load_model()
    core.BANDWIDTH = 24.0
    out: dict = {"by_scheme": {}}
    agg = {k: [] for k in ("conv13", "conv12", "tb13")}
    perm_tb = tb_perm(KEY)

    print("自检：咬尾编解码闭环", flush=True)
    probe = np.random.default_rng(7).integers(0, 2, size=core.PAYLOAD_BITS, dtype=np.int8)
    loop = tb_decode(tb_encode(probe, 3), 3, core.PAYLOAD_BITS)
    print(f"  noiseless roundtrip exact = {np.array_equal(loop, probe)}", flush=True)

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

            for scheme in ("conv13", "conv12", "tb13"):
                rows = []
                for secret in SECRETS[:per_clip]:
                    message = core.hex_to_bits(secret)
                    if scheme == "tb13":
                        coded = tb_encode(message, 3)
                        perm = perm_tb
                        n_bits = core.PAYLOAD_BITS * 3
                    else:
                        coded, perm = core.encode_message(message, scheme, KEY)
                        n_bits = core.coded_length(scheme)
                    coded = np.asarray(coded, dtype=np.int8)
                    positions = np.linspace(0, frames - 1, n_bits, dtype=np.int64)
                    changed = positions[bit_map[tokens[positions]] != coded]
                    modified = tokens.copy()
                    modified[changed] = partner[tokens[changed]]
                    stego = cover_codes.clone()
                    stego[0, 0, core.LAYER] = torch.from_numpy(modified)
                    wave = model.decode(stego, encoded.audio_scales, None).audio_values.squeeze().cpu().numpy()
                    pcm, _ = core._pcm16_roundtrip(wave)
                    re_tokens = model.encode(core._signal_tensor(pcm), bandwidth=core.BANDWIDTH).audio_codes
                    received = bit_map[re_tokens[0, 0, core.LAYER].cpu().numpy().astype(np.int64)[positions]]
                    if scheme == "tb13":
                        bits = np.empty_like(received)
                        bits[perm] = received
                        decoded = tb_decode(bits, 3, core.PAYLOAD_BITS)
                    else:
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
            print(
                f"{path.name[:18]:18s} {info['duration_seconds']:5.2f}s  "
                + "  ".join(
                    f"{s}: {np.mean([r['exact'] for r in agg[s][-per_clip:]])*100:5.1f}%/"
                    f"{np.mean([r['changed'] for r in agg[s][-per_clip:]]):4.1f}"
                    for s in ("conv13", "conv12", "tb13")
                ),
                flush=True,
            )

    print("\n== 汇总（exact% / 平均改动数 / 分段SNR）==")
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
            f"  {scheme:6s} exact={summary['exact']*100:5.1f}% changed={summary['changed']:5.1f} "
            f"raw={summary['raw']:.3f} seg={summary['seg']:5.2f} lsd={summary['lsd']:4.2f}",
            flush=True,
        )

    target = HERE / "runtime" / "tailbite_probe.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}", flush=True)


if __name__ == "__main__":
    main()
