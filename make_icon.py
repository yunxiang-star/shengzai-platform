# -*- coding: utf-8 -*-
"""生成演示台图标：深色底 + 青色声波 + 右下角小锁。

要点：不同尺寸**单独绘制**。16px 里塞 7 根声波只会糊成一团，
所以小尺寸减少根数、加粗笔画，保证任务栏/桌面小图标也认得出。

输出 assets/app.ico（16–256 多尺寸）与 assets/app.png（256 预览）。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ICO = ASSETS / "app.ico"
PNG = ASSETS / "app.png"

SS = 4  # 超采样倍数，用于抗锯齿


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def gradient(size: int) -> Image.Image:
    top, bottom = (10, 26, 42), (19, 56, 84)
    strip = Image.new("RGB", (1, size))
    for y in range(size):
        strip.putpixel((0, y), lerp(top, bottom, y / max(size - 1, 1)))
    return strip.resize((size, size))


def render(size: int) -> Image.Image:
    """按目标尺寸原生绘制（内部 4x 超采样）。"""
    s = size * SS
    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # 圆角底
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, s - 1, s - 1), radius=int(s * 0.22), fill=255
    )
    canvas.paste(gradient(s), (0, 0), mask)

    draw = ImageDraw.Draw(canvas)

    if size <= 20:
        heights = [0.40, 0.95, 0.60]
        bar_ratio, gap_ratio = 0.115, 0.055
        center_x, center_y = 0.40, 0.44
    elif size <= 36:
        heights = [0.26, 0.58, 0.95, 0.66, 0.34]
        bar_ratio, gap_ratio = 0.098, 0.042
        center_x, center_y = 0.40, 0.44
    else:
        heights = [0.18, 0.38, 0.64, 0.94, 0.70, 0.46, 0.22]
        bar_ratio, gap_ratio = 0.066, 0.032
        center_x, center_y = 0.42, 0.44

    n = len(heights)
    bar_w = s * bar_ratio
    gap = s * gap_ratio
    total = n * bar_w + (n - 1) * gap
    x = s * center_x - total / 2
    cy = s * center_y
    for h in heights:
        bar_h = s * 0.60 * h
        draw.rounded_rectangle(
            (x, cy - bar_h / 2, x + bar_w, cy + bar_h / 2),
            radius=bar_w / 2,
            fill=(56, 225, 200, 255),
        )
        x += bar_w + gap

    # 锁徽章：小尺寸也要有厚度，否则只剩一个黄点
    cx, cy2 = s * 0.755, s * 0.755
    r = s * (0.150 if size <= 20 else 0.138)
    draw.ellipse((cx - r * 1.30, cy2 - r * 1.30, cx + r * 1.30, cy2 + r * 1.30),
                 fill=(255, 196, 77, 255))
    body = (cx - r * 0.74, cy2 - r * 0.10, cx + r * 0.74, cy2 + r * 0.82)
    draw.rounded_rectangle(body, radius=r * 0.20, fill=(22, 38, 56, 255))
    draw.arc(
        (cx - r * 0.48, cy2 - r * 0.86, cx + r * 0.48, cy2 + r * 0.18),
        start=180, end=360, fill=(22, 38, 56, 255), width=max(int(r * 0.26), 2),
    )
    draw.rectangle(
        (cx - r * 0.48, cy2 - r * 0.32, cx + r * 0.48, cy2 - r * 0.04),
        fill=(255, 196, 77, 255),
    )
    if size > 20:
        draw.ellipse(
            (cx - r * 0.15, cy2 + r * 0.18, cx + r * 0.15, cy2 + r * 0.50),
            fill=(255, 196, 77, 255),
        )

    return canvas.resize((size, size), Image.LANCZOS)


def build_ico(frames: list[Image.Image]) -> bytes:
    """手工拼 ICO 容器。

    PIL 的 ICO 保存会用单张基准图重算所有尺寸，把按尺寸单独绘制的差异抹掉；
    这里直接写 ICONDIR + ICONDIRENTRY + 每帧 PNG 数据（Vista+ 支持 PNG 条目）。
    """
    import io
    import struct

    blobs = []
    for frame in frames:
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG", optimize=True)
        blobs.append(buffer.getvalue())

    header = struct.pack("<HHH", 0, 1, len(frames))
    offset = 6 + 16 * len(frames)
    entries = b""
    for frame, blob in zip(frames, blobs):
        side = frame.width
        entries += struct.pack(
            "<BBBBHHII",
            0 if side >= 256 else side,   # 256 用 0 表示
            0 if side >= 256 else side,
            0, 0, 1, 32, len(blob), offset,
        )
        offset += len(blob)
    return header + entries + b"".join(blobs)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [render(s) for s in sizes]
    ICO.write_bytes(build_ico(frames))
    frames[-1].save(PNG)
    print(f"icon written: {ICO}  sizes={sizes}")


if __name__ == "__main__":
    main()
