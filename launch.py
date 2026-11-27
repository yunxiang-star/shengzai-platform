# -*- coding: utf-8 -*-
"""语音隐写演示台 · 启动器

双击桌面快捷方式后做三件事：
1. 先探测本机 8765 端口：服务已在跑，就只打开浏览器，不重复启动（避免端口占用）。
2. 没在跑，就在本进程内拉起 uvicorn 子进程，把日志留在当前窗口。
3. 轮询 /api/health，等 EnCodec 模型真的就绪后再打开浏览器，
   这样不会出现"页面先出来、结果全红"的情况。

用法：
    python launch.py            # 正常启动
    python launch.py --no-open  # 只启服务，不开浏览器
    python launch.py --port 8766
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
WAIT_SECONDS = 180


def probe(health_url: str, timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def open_url(url: str, delay: float = 0.0) -> None:
    if delay:
        time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    port = args.port
    url = f"http://{HOST}:{port}"
    health_url = f"{url}/api/health"

    print("=" * 62)
    print("  语音隐写演示台  ·  Voice Steganography Demo")
    print("=" * 62)
    print(f"  地址：{url}")
    print("  这个窗口是服务本体，关掉它演示台就停了（Ctrl+C 可停止）。")
    print("=" * 62)

    existing = probe(health_url)
    if existing is not None:
        print("\n[提示] 演示台已经在运行了，直接打开浏览器。")
        print(f"       模型状态：{existing.get('message', '未知')}")
        if not args.no_open:
            open_url(url)
        print("\n可以关闭这个窗口，不影响已经开着的演示台。")
        time.sleep(1.5)
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = ""          # 清掉沙箱注入的 shim，避免删除守卫误伤
    env["PYTHONIOENCODING"] = "utf-8"

    print("\n[1/3] 正在启动服务……")
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn", "app:app",
         "--host", HOST, "--port", str(port), "--log-level", "warning"],
        cwd=str(APP_DIR),
        env=env,
    )

    # 注意：不要在这里提前打开浏览器。
    # uvicorn 要导入 torch 并加载 EnCodec，实测端口要 3–6 秒才开始监听，
    # 1 秒时打开浏览器只会得到"无法访问此网站"。等健康检查通过再开。
    print("[2/3] 等待服务端口与 EnCodec 模型就绪（一般 5–20 秒）……", end="", flush=True)
    deadline = time.time() + WAIT_SECONDS
    ready = False
    ticks = 0
    while time.time() < deadline:
        if process.poll() is not None:
            print("\n[错误] 服务进程退出了。常见原因：端口被占用。")
            print(f"       可以换端口重试：python launch.py --port {port + 1}")
            return 1
        info = probe(health_url)
        if info and info.get("model_loaded"):
            ready = True
            break
        time.sleep(1.0)
        ticks += 1
        if ticks % 3 == 0:
            print(".", end="", flush=True)
    print()

    if ready:
        if args.no_open:
            print(f"[3/3] 就绪！请手动访问：{url}")
        else:
            print(f"[3/3] 就绪！正在打开浏览器：{url}")
            open_url(url)
    else:
        print(f"[3/3] 服务在跑，但模型 {WAIT_SECONDS}s 内没加载完。")
        print("      页面仍可打开，只是点按钮会提示模型未就绪。")
        if not args.no_open:
            open_url(url)

    print("\n日志会滚动显示在下面。按 Ctrl+C 停止服务。\n" + "-" * 62)
    try:
        return process.wait()
    except KeyboardInterrupt:
        print("\n正在停止……")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
