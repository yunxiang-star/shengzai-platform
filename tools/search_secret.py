# -*- coding: utf-8 -*-
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import core

wav, _ = core.read_audio_bytes((APP / "assets" / "demo.wav").read_bytes())
for secret in ["A7C3", "1234", "5A3C", "0F0F", "DEAD", "BEEF", "2468", "ACE0", "1357", "9BDF", "C0DE", "3C5A"]:
    result = core.analyze_embedding(wav, "hpp", core.hex_to_bits(secret), "demo-2026")
    print(secret, result["exact"], result["recovered_hex"],
          round(result["raw_bit_accuracy"], 3), round(result["stoi"], 3),
          round(result["receiver_hist_l1"], 3), flush=True)
