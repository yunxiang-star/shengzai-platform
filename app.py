# -*- coding: utf-8 -*-
"""语音隐写演示台：FastAPI 后端。"""
from __future__ import annotations

import json
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import core

APP_DIR = Path(__file__).resolve().parent
RUNTIME = APP_DIR / "runtime"
STATIC = APP_DIR / "static"
ASSETS = APP_DIR / "assets"
RUNTIME.mkdir(exist_ok=True)
STATIC.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        core.load_model()
    except Exception:
        # 健康检查会显示模型未就绪；真正请求时还会再尝试一次。
        traceback.print_exc()
    yield


app = FastAPI(title="语音隐写演示台", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")
app.mount("/outputs", StaticFiles(directory=str(RUNTIME)), name="outputs")


def validate_mode_key(mode: str, key: str):
    if mode not in {"direct", "hpp"}:
        raise HTTPException(400, detail="模式只能是 direct 或 hpp。")
    key = (key or "").strip()
    if len(key) < 4:
        raise HTTPException(400, detail="演示密钥至少 4 个字符。")
    if len(key) > 64:
        raise HTTPException(400, detail="演示密钥太长，请控制在 64 字符以内。")
    return key


def public_payload(result, session: str, info: dict):
    return {
        "session": session,
        "mode": result["mode"],
        "mode_label": "直接替换索引" if result["mode"] == "direct" else "直方图保持置换（HPP）",
        "scheme": result["scheme"],
        "attempts": result.get("attempts", 1),
        "max_attempts": result.get("max_attempts", 1),
        "message_hex": result["message_hex"],
        "message_bits": result["message_bits"],
        "ascii_hint": result["ascii_hint"],
        "recovered_hex": result["recovered_hex"],
        "recovered_ascii_hint": result["recovered_ascii_hint"],
        "exact": result["exact"],
        "raw_bit_accuracy": result["raw_bit_accuracy"],
        "wrong_key": result["wrong_key"],
        "stoi": result["stoi"],
        "perceptual": result.get("perceptual"),
        "perceptual_source": result.get("perceptual_source"),
        "pairing": result.get("pairing"),
        "snr_db": result["snr_db"],
        "sender_hist_l1": result["sender_hist_l1"],
        "receiver_hist_l1": result["receiver_hist_l1"],
        "changed_ratio": result["changed_ratio"],
        "changed_tokens": result["changed_tokens"],
        "carrier_positions": result["carrier_positions"],
        "frame_count": result["frame_count"],
        "bandwidth_kbps": result.get("bandwidth_kbps"),
        "quantizers": result.get("quantizers"),
        "payload_bits": result.get("payload_bits"),
        "payload_bps": result["payload_bps"],
        "duration_seconds": info["duration_seconds"],
        "truncated": info["truncated"],
        "token_preview": result["token_preview"],
        "files": {
            "source": f"/outputs/{session}/source.wav",
            "cover": f"/outputs/{session}/cover.wav",
            "stego": f"/outputs/{session}/stego.wav",
            "waveform": f"/outputs/{session}/waveform.png",
            "histogram": f"/outputs/{session}/histogram.png",
            "divergence": f"/outputs/{session}/divergence.png",
        },
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_loaded": core.MODEL is not None,
        "sr": core.SR,
        "bandwidth_kbps": core.BANDWIDTH,
        "payload_bits": core.PAYLOAD_BITS,
        "device": "cpu",
        "message": "模型已就绪" if core.MODEL is not None else "模型尚未加载",
    }


@app.post("/api/embed")
def embed(
    audio: UploadFile = File(...),
    mode: str = Form("direct"),
    secret: str = Form("C0"),
    key: str = Form("demo-2026"),
    nonce: str = Form(""),
    bandwidth: float = Form(core.BANDWIDTH),
    payload: int = Form(core.PAYLOAD_BITS),
):
    mode = (mode or "direct").strip()
    key = validate_mode_key(mode, key)
    try:
        bandwidth = core.normalize_bandwidth(bandwidth)
        payload = core.normalize_payload(payload)
        message = core.hex_to_bits(secret, payload)
        raw = audio.file.read()
        wav, info = core.read_audio_bytes(raw)
        session = uuid.uuid4().hex[:12]
        session_dir = RUNTIME / session

        result = core.analyze_embedding(
            wav, mode, message, key, nonce=nonce[:32], bandwidth=bandwidth, payload_bits=payload
        )
        fallback_from = None
        if not result["exact"]:
            # 宁可改路子也不交付一个读不回来的样本，但必须如实标注换了哪条路。
            other = "hpp" if mode == "direct" else "direct"
            alt = core.analyze_embedding(
                wav, other, message, key, nonce=nonce[:32], bandwidth=bandwidth, payload_bits=payload
            )
            if alt["exact"]:
                fallback_from = mode
                result = alt
        if not result["exact"]:
            suggestion = (
                "请把口令长度切到 8 bit 后重试。"
                if payload == 16
                else "请换一段更长、静音更少的语音后重试。"
            )
            raise HTTPException(
                422,
                detail=f"这段音频没有通过发送前的完整提取自检，未生成交付文件。{suggestion}",
            )
        session_dir.mkdir(parents=True, exist_ok=False)
        core.save_wav(session_dir / "source.wav", wav)
        core.save_wav(session_dir / "cover.wav", result["cover_wave"])
        core.save_wav(session_dir / "stego.wav", result["stego_wave"])
        core.render_waveform(
            session_dir / "waveform.png", result["cover_wave"], result["stego_wave"], wav
        )
        core.render_histograms(session_dir / "histogram.png", result)
        core.render_histogram_divergence(session_dir / "divergence.png", result)

        payload = public_payload(result, session, info)
        payload["fallback_from"] = fallback_from
        if fallback_from == "hpp":
            payload["fallback_note"] = (
                f"你选的「换座位，不换书单」在这段音频上试了 {core.HPP_MAX_ATTEMPTS} 种合法置换"
                "仍读不回来，已自动改用「直接换抽屉」。这条能读回，但索引直方图痕迹更明显。"
            )
        elif fallback_from == "direct":
            payload["fallback_note"] = (
                "「直接换抽屉」这次纠错后仍差几位，已自动改用「换座位，不换书单」交付。"
                "这条能读回、直方图痕迹更小，但它本身也有失败率，只是这次成功了。"
            )
        else:
            payload["fallback_note"] = None
        (session_dir / "metadata.json").write_text(
            json.dumps({**payload, "key_fingerprint": core.key_seed(key, "fingerprint") % 100000},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload
    except core.EmbeddingInfeasible as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except core.DemoError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            500,
            detail="这次演示在服务器内部失败了。请换一段 2–12 秒的 WAV 再试；详细信息已写入终端。",
        ) from exc


@app.post("/api/extract")
def extract(
    audio: UploadFile = File(...),
    mode: str = Form("direct"),
    key: str = Form("demo-2026"),
    expected: str = Form(""),
    bandwidth: float = Form(core.BANDWIDTH),
    payload: int = Form(core.PAYLOAD_BITS),
):
    mode = (mode or "direct").strip()
    key = validate_mode_key(mode, key)
    try:
        bandwidth = core.normalize_bandwidth(bandwidth)
        payload = core.normalize_payload(payload)
        raw = audio.file.read()
        wav, info = core.read_audio_bytes(raw)
        result = core.extract_file(wav, mode, key, bandwidth=bandwidth, payload_bits=payload)
        expected_clean = (expected or "").strip().upper().replace(" ", "")
        expected_result = None
        if expected_clean:
            expected_bits = core.hex_to_bits(expected_clean, payload)
            expected_result = {
                "expected_hex": expected_clean,
                "exact": bool(np.array_equal(result["bits"], expected_bits)),
                "bit_accuracy": float(np.mean(result["bits"] == expected_bits)),
            }
        return {
            "mode": mode,
            "mode_label": "直接替换索引" if mode == "direct" else "直方图保持置换（HPP）",
            "scheme": result["scheme"],
            "message_hex": result["hex"],
            "message_bits": result["bits"].tolist(),
            "ascii_hint": result["ascii_hint"],
            "positions": result["positions"],
            "frame_count": result["frame_count"],
            "bandwidth_kbps": result.get("bandwidth_kbps"),
            "payload_bits": payload,
            "duration_seconds": round(info["duration_seconds"], 2),
            "expected": expected_result,
        }
    except core.DemoError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            500,
            detail="提取失败。请确认音频是本站生成的、模式与密钥一致，并保持 24 kHz EnCodec 链路。",
        ) from exc
