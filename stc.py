"""Syndrome-Trellis Codes（STC）最小失真嵌入，纯 numpy 实现。

问题形式：给定 n 个二进制位置、逐位置代价 rho_i（改动该位的"疼"），
以及 k 位目标 syndrome，求改动向量 e ∈ {0,1}^n，使

    H e = syndrome  (mod 2)     且     Σ rho_i · e_i 最小

H 由密钥决定：k 位拆成 nb 块、每块 h 位（k = nb·h），块 b 占用列区间
[b·w, (b+1)·w)，块内第 i 列是一个随机 h 位非零向量。块之间互不重叠，
因此每块可用 2^h 状态的动态规划精确求解，复杂度 O(w · 2^h)。

注意：这也正是 STC 在"擦除信道"上的软肋 —— 一个通道错误会污染约 h/2 个
syndrome 位，而这 h/2 位全在同一个 h 位块里。见 tools/stc_probe.py 的实测。
"""
from __future__ import annotations

import numpy as np

INF = 1e18


def build_columns(seed: int, h: int, w: int, nb: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 1 << h, size=(nb, w), dtype=np.int64)


def syndrome_of(bits: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """计算 H x：每块把该块内取值为 1 的列异或起来。"""
    nb, w = columns.shape
    out = np.zeros(nb, dtype=np.int64)
    bits = np.asarray(bits).astype(np.int64)
    for b in range(nb):
        block = bits[b * w : (b + 1) * w]
        out[b] = int(np.bitwise_xor.reduce(columns[b][block == 1], initial=0))
    return out


def embed(
    cover_bits: np.ndarray,
    costs: np.ndarray,
    target: np.ndarray,
    columns: np.ndarray,
    h: int,
):
    """最小代价地满足 H e = target ⊖ H cover。

    参数
    ----
    cover_bits : (n,) 0/1
    costs      : (n,) 每个位置被改动的代价（≥0）
    target     : (nb,) 每块的 h 位目标整数
    columns    : (nb, w) 每块的列向量
    h          : 每块 syndrome 位数

    返回 (e, total_cost)，e 为 (n,) bool。
    """
    nb, w = columns.shape
    states = 1 << h
    index = np.arange(states)
    e = np.zeros((nb, w), dtype=np.int8)
    total = 0.0
    for b in range(nb):
        dp = np.full(states, INF)
        dp[0] = 0.0
        backs = np.zeros((w, states), dtype=bool)
        for i in range(w):
            col = int(columns[b, i])
            stay = dp
            take = dp + float(costs[b * w + i])
            reached = np.full(states, INF)
            np.minimum.at(reached, index ^ col, take)
            use = reached < stay
            backs[i] = use
            dp = np.where(use, reached, stay)
        goal = int(target[b])
        if not np.isfinite(dp[goal]):
            raise RuntimeError(f"block {b}: syndrome unreachable")
        total += float(dp[goal])
        state = goal
        for i in range(w - 1, -1, -1):
            if backs[i, state]:
                e[b, i] = 1
                state ^= int(columns[b, i])
        if state != 0:
            raise RuntimeError(f"block {b}: backtrack mismatch")
    return e.reshape(-1).astype(bool), total
