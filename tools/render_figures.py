# -*- coding: utf-8 -*-
"""把 index.html 里的 SVG 渲染成 PNG，用于检查排版（不能开浏览器时的替代手段）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

APP = Path(__file__).resolve().parents[1]
OUT = APP / "runtime" / "svg_preview"
OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    ("Microsoft YaHei", r"C:/Windows/Fonts/msyh.ttc"),
    ("SimHei", r"C:/Windows/Fonts/simhei.ttf"),
]


def register_fonts():
    for name, path in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=0))
            print(f"font ok: {name}")
            return name
        except Exception as exc:
            print(f"font fail: {name} {exc}")
    return None


def main():
    register_fonts()
    html = (APP / "static" / "index.html").read_text(encoding="utf-8")
    svgs = re.findall(r"<svg[\s\S]*?</svg>", html)
    print("found", len(svgs))
    for index, svg in enumerate(svgs, 1):
        # reportlab 不认识 marker，先去掉箭头标记，避免渲染报错
        cleaned = re.sub(r"<defs>[\s\S]*?</defs>", "", svg)
        cleaned = cleaned.replace(' marker-end="url(#ar)"', "")
        # 浏览器里靠 CSS 兜底；这里的渲染器只认已注册的字体名，去掉 fallback 才能画出中文
        cleaned = re.sub(r'font-family="[^"]*"', 'font-family="Microsoft YaHei"', cleaned)
        target = OUT / f"fig{index}.svg"
        target.write_text(cleaned, encoding="utf-8")
        try:
            drawing = svg2rlg(str(target))
            if drawing is None:
                print(f"fig{index}: render returned None")
                continue
            scale = 2.0
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)
            png = OUT / f"fig{index}.png"
            renderPM.drawToFile(drawing, str(png), fmt="PNG", bg=0xFFFFFF)
            print(f"fig{index}: {png.name} {drawing.width:.0f}x{drawing.height:.0f}")
        except Exception as exc:
            print(f"fig{index}: FAILED {type(exc).__name__} {exc}")


if __name__ == "__main__":
    sys.exit(main())
