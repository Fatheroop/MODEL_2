#!/usr/bin/env python3
"""
===================================================================================================
ORGANISM: CORE COGNITIVE MODEL (numpy + Pillow only, zero backprop)
===================================================================================================
A biologically grounded perception -> memory -> motivation -> action cognitive model that learns
continuously from sensory experience without backpropagation, loss gradients, or autograd.

Core Architectural Modules:
  * Retina: Biological receptive field visual encoder (6 channels: luminance, saturation, motion,
    horizontal/vertical gradients, local neighborhood contrast; pooled scene representation; local features).
  * SparseEncoder: k-winners-take-all competitive Hebbian layer with homeostatic thresholds.
  * EpisodicMemory: Engram prototype memory with novelty scoring and sleep consolidation.
  * ForwardModel: Predictive sensorimotor forward model predicting latent gist and learning progress.
  * TDHebbianPolicy: TD(lambda) three-factor Hebbian reinforcement learning with eligibility traces.
  * LanguageCortex: Hebbian N-gram cortex with surprise-gated learning, homeostatic decay, and inner speech.
  * Drives: Homeostatic drives (curiosity, boredom, fatigue, social).
  * GlobalWorkspace: Working memory stage with salience competition and broadcast.
  * SelfModel: Metacognitive self-monitoring (running reward, error, confidence, behavioural entropy).
  * Organism: Unified cognitive agent integrating perception, memory, drives, policy, and language.
===================================================================================================
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import random
import re
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# =================================================================================================
# GEOMETRY / ACTION SPACE
# =================================================================================================
GRID_W, GRID_H = 64, 36                 # retina resolution
CELL_COLS, CELL_ROWS = 8, 4             # fixation targets: 8x4 grid over the content area
N_CELLS = CELL_COLS * CELL_ROWS         # 32 click targets
A_SCROLL_DOWN = N_CELLS
A_SCROLL_UP = N_CELLS + 1
A_READ = N_CELLS + 2                    # dwell on the screen and read it
A_TYPE = N_CELLS + 3                    # type one short learned word + Enter
A_BACK = N_CELLS + 4                    # navigation "back"
N_ACTIONS = N_CELLS + 5
N_LOC = 8                               # local features per fixation cell
X_DIM = 6 * (GRID_H // 4) * (GRID_W // 4) + 6   # pooled scene vector (864 + 6 colour means)
CKPT_DIR = os.path.abspath("checkpoints")


def action_name(a: int) -> str:
    if a < N_CELLS:
        return f"click({a % CELL_COLS},{a // CELL_COLS})"
    return {A_SCROLL_DOWN: "scroll_down", A_SCROLL_UP: "scroll_up", A_READ: "read",
            A_TYPE: "type", A_BACK: "back"}[a]


# =================================================================================================
# PERCEPTION: RETINA
# =================================================================================================
class Retina:
    """64x36 retina -> 6 feature channels; scene vector for memory; 8 local features for each of the 32 fixation cells."""

    def __init__(self):
        self.mu = np.zeros(N_LOC, np.float32)
        self.var = np.ones(N_LOC, np.float32)
        self.n = 0

    @staticmethod
    def lum_of(rgb: np.ndarray) -> np.ndarray:
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    def encode(self, rgb: np.ndarray, motion: np.ndarray) -> np.ndarray:
        lum = self.lum_of(rgb)
        sat = rgb.max(-1) - rgb.min(-1)
        gx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
        gy = np.abs(np.diff(lum, axis=0, append=lum[-1:, :]))
        nb = (np.roll(lum, 1, 0) + np.roll(lum, -1, 0) + np.roll(lum, 1, 1) + np.roll(lum, -1, 1)) / 4.0
        m6 = np.stack([lum, sat, np.clip(motion, 0, 1), gx, gy, np.abs(lum - nb)], axis=-1)
        return m6.astype(np.float32)

    def pooled(self, m6: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Layout + colour-scheme signature of the scene (unit norm). Motion is excluded: it is not identity."""
        by = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
        maps = [m6[..., 0], m6[..., 1], m6[..., 3], m6[..., 4], m6[..., 5], by]
        parts, means = [], []
        for mp in maps:
            p = mp.reshape(GRID_H // 4, 4, GRID_W // 4, 4).mean(axis=(1, 3))
            means.append(float(p.mean()))
            parts.append(((p - p.mean()) / (p.std() + 0.05)).ravel())
        v = np.concatenate(parts + [np.array(means, np.float32) * 3.0]).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def cell_features(self, m6: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        lum = m6[..., 0]
        by = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
        ch, cw = GRID_H // CELL_ROWS, GRID_W // CELL_COLS
        base = m6.reshape(CELL_ROWS, ch, CELL_COLS, cw, 6).mean(axis=(1, 3)).reshape(N_CELLS, 6)
        lstd = lum.reshape(CELL_ROWS, ch, CELL_COLS, cw).std(axis=(1, 3)).reshape(N_CELLS, 1)
        byc = by.reshape(CELL_ROWS, ch, CELL_COLS, cw).mean(axis=(1, 3)).reshape(N_CELLS, 1)
        raw = np.concatenate([base, lstd, byc], axis=1).astype(np.float32)
        a = 0.05 if self.n < 50 else 0.005
        self.mu += a * (raw.mean(0) - self.mu)
        self.var += a * (raw.var(0) - self.var)
        self.n += 1
        return np.clip((raw - self.mu) / np.sqrt(self.var + 1e-4), -3, 3).astype(np.float32) * 0.5


# =================================================================================================
# CORTEX: SPARSE COMPETITIVE CODE, EPISODIC MEMORY, FORWARD MODEL
# =================================================================================================
class SparseEncoder:
    """k-winners-take-all competitive Hebbian layer with homeostatic thresholds (local rule, no gradients)."""

    def __init__(self, n_in: int = X_DIM, n_units: int = 2048, k: int = 24, eta: float = 0.05, seed: int = 1):
        rng = np.random.RandomState(seed)
        W = rng.randn(n_units, n_in).astype(np.float32)
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.bias = np.zeros(n_units, np.float32)
        self.rate = np.full(n_units, k / n_units, np.float32)
        self.n_units, self.k, self.eta = n_units, k, eta

    def encode(self, x: np.ndarray, learn: bool = True) -> np.ndarray:
        pot = self.W @ x + self.bias
        idx = np.argpartition(pot, -self.k)[-self.k:]
        if learn:
            w = self.W[idx]
            w += self.eta * (x[None, :] - w)
            self.W[idx] = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
            fired = np.zeros(self.n_units, np.float32)
            fired[idx] = 1.0
            self.rate += 0.002 * (fired - self.rate)
            self.bias = -1.5 * (self.rate - self.k / self.n_units)       # over-used units get harder to fire
        return idx


class EpisodicMemory:
    """Prototype memory of scenes seen. Gives novelty, and per-context action values (one-shot recall)."""

    def __init__(self, dim: int = X_DIM, capacity: int = 600, alloc: float = 0.30):
        self.P = np.zeros((capacity, dim), np.float32)
        self.count = np.zeros(capacity, np.float32)
        self.last = np.zeros(capacity, np.float32)
        self.Qe = np.zeros((capacity, N_ACTIONS), np.float32)
        self.n, self.cap, self.alloc = 0, capacity, alloc

    def observe(self, x: np.ndarray, step: int) -> Tuple[int, float, float]:
        if self.n == 0:
            self._put(0, x, step)
            self.n = 1
            return 0, 1.0, 0.0
        sims = self.P[:self.n] @ x
        j = int(np.argmax(sims))
        sim = float(sims[j])
        novelty = float(np.clip((1.0 - sim - 0.10) / 0.5, 0.0, 1.0))
        if 1.0 - sim > self.alloc:
            if self.n < self.cap:
                j = self.n
                self.n += 1
            else:                                        # forget the least-used, least-recent scene
                j = int(np.argmin(self.count[:self.n] + 1e-4 * self.last[:self.n]))
            self._put(j, x, step)
        else:
            p = self.P[j] + 0.1 * (x - sim * self.P[j])
            self.P[j] = p / (np.linalg.norm(p) + 1e-8)
            self.count[j] += 1
            self.last[j] = step
        return j, novelty, sim

    def _put(self, j: int, x: np.ndarray, step: int):
        self.P[j], self.count[j], self.last[j], self.Qe[j] = x, 1.0, step, 0.0

    def credit(self, j: int, a: int, r: float):
        self.Qe[j, a] += 0.3 * (r - self.Qe[j, a])

    def consolidate(self, thresh: float = 0.96) -> int:
        """Sleep-time merge of near-duplicate prototypes. Returns number merged."""
        n, merged, dead = self.n, 0, set()
        S = self.P[:n] @ self.P[:n].T
        for i in range(n):
            if i in dead:
                continue
            for j in range(i + 1, n):
                if j not in dead and S[i, j] > thresh:
                    keep, drop = (i, j) if self.count[i] >= self.count[j] else (j, i)
                    self.count[keep] += self.count[drop]
                    self.Qe[keep] = 0.5 * (self.Qe[keep] + self.Qe[drop])
                    dead.add(drop)
                    merged += 1
                    if drop == i:
                        break
        if dead:
            keep_idx = [i for i in range(n) if i not in dead]
            m = len(keep_idx)
            for arr in (self.P, self.count, self.last, self.Qe):
                arr[:m] = arr[keep_idx]
            self.n = m
        return merged


class ForwardModel:
    """Predicts the next scene gist from (gist, action) with a local delta rule. Error = surprise."""

    def __init__(self, dim_g: int = 64, seed: int = 2):
        rng = np.random.RandomState(seed)
        self.R = (rng.randn(dim_g, X_DIM) / math.sqrt(X_DIM)).astype(np.float32)
        self.M = np.zeros((dim_g, dim_g + N_ACTIONS), np.float32)
        self.fast, self.slow = 0.15, 0.15

    def gist(self, x: np.ndarray) -> np.ndarray:
        g = self.R @ x
        return g / (np.linalg.norm(g) + 1e-8)

    def update(self, g: np.ndarray, a: int, g_next: np.ndarray, eta: float = 0.03) -> float:
        z = np.zeros(self.M.shape[1], np.float32)
        z[:g.size] = g
        z[g.size + a] = 1.0
        e = g_next - self.M @ z
        self.M += eta * np.outer(e, z)
        err = float(np.mean(np.abs(e)))
        self.fast += 0.3 * (err - self.fast)
        self.slow += 0.02 * (err - self.slow)
        return err

    @property
    def progress(self) -> float:                         # learning progress: error falling = interesting
        return self.slow - self.fast


# =================================================================================================
# POLICY: TD(lambda) THREE-FACTOR HEBBIAN
# =================================================================================================
class TDHebbianPolicy:
    """
    Q(s,a) = sum_{i in active code} W[a,i]  +  V . f_cell(a)           (click actions share V -> generalises)
                                            +  U[a] . f_global         (scroll / read / type / back)
    delta = r + gamma * max Q(s') - Q(s,a);   w += eta * delta * eligibility
    It is a one-layer linear learner (no gradient flows through layers, no loss function, no autograd).
    """

    def __init__(self, n_units: int, k: int, gamma: float = 0.9, lam: float = 0.6,
                 eta: float = 0.15, eta_v: float = 0.05):
        self.W = np.zeros((N_ACTIONS, n_units), np.float32)
        self.V = np.zeros(N_LOC, np.float32)
        self.U = np.zeros((N_ACTIONS - N_CELLS, N_LOC), np.float32)
        self.eW, self.eV, self.eU = np.zeros_like(self.W), np.zeros_like(self.V), np.zeros_like(self.U)
        self.gamma, self.lam, self.eta, self.eta_v, self.k = gamma, lam, eta, eta_v, k
        self.last_delta = 0.0

    def q(self, idx: np.ndarray, F: np.ndarray, gf: np.ndarray) -> np.ndarray:
        q = self.W[:, idx].sum(axis=1)
        q[:N_CELLS] += F @ self.V
        q[N_CELLS:] += self.U @ gf
        return q

    @staticmethod
    def select(q: np.ndarray, eps: float, tau: float, rng: np.random.RandomState) -> int:
        if rng.rand() < eps:
            return int(rng.randint(N_ACTIONS))
        z = (q - q.max()) / max(tau, 1e-3)
        p = np.exp(np.clip(z, -30, 0))
        p /= p.sum()
        return int(rng.choice(N_ACTIONS, p=p))

    def update(self, st: Tuple[np.ndarray, np.ndarray, np.ndarray], a: int, r: float,
               st_next: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        idx, F, gf = st
        q_sa = float(self.q(idx, F, gf)[a])
        delta = r + self.gamma * float(self.q(*st_next).max()) - q_sa
        self.last_delta = delta
        gl = self.gamma * self.lam
        self.eW *= gl
        self.eV *= gl
        self.eU *= gl
        self.eW[a, idx] += 1.0
        if a < N_CELLS:
            self.eV += F[a]
        else:
            self.eU[a - N_CELLS] += gf
        self.W += (self.eta / self.k) * delta * self.eW
        self.V += self.eta_v * delta * self.eV
        self.U += self.eta_v * delta * self.eU
        np.clip(self.W, -4, 4, out=self.W)
        np.clip(self.V, -4, 4, out=self.V)
        np.clip(self.U, -4, 4, out=self.U)

    def replay_update(self, st: Tuple[np.ndarray, np.ndarray, np.ndarray], a: int, r: float,
                      st_next: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Offline (sleep) one-step update, no traces."""
        idx, F, gf = st
        delta = r + self.gamma * float(self.q(*st_next).max()) - float(self.q(idx, F, gf)[a])
        self.W[a, idx] += (self.eta / self.k) * delta
        if a < N_CELLS:
            self.V += self.eta_v * delta * F[a]
        else:
            self.U[a - N_CELLS] += self.eta_v * delta * gf


# =================================================================================================
# LANGUAGE: HEBBIAN N-GRAM CORTEX
# =================================================================================================
class LanguageCortex:
    FUNC = frozenset("the a an of in to and is was that for on with as by at from it can cannot could would should "
                     "will have has had he she they we i you her his my our their so but or one are be this these "
                     "those not no been were which also its than then".split())
    BOILER = ("jump to", "navigation", "cookie", "privacy policy", "terms of use", "disclaimer", "contact us",
              "donate", "view history", "edit this", "download as", "sign in", "log in", "subscribe",
              "all rights reserved", "main menu", "toggle", "skip to")

    def __init__(self):
        self.w2i: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.i2w: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.count: Dict[int, int] = {}
        self.bi: Dict[int, Dict[int, float]] = {}
        self.bi_tot: Dict[int, float] = {}
        self.tri: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.co: Dict[int, Dict[int, float]] = {}
        self.eta = 0.25

    @staticmethod
    def words_of(text: str) -> List[str]:
        return re.findall(r"[a-z][a-z0-9']*", text.lower())

    def sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s*\n\s*", " ", text)
        out = []
        for s in re.split(r"(?<=[.!?])\s+", text):
            toks = s.split()
            if not 4 <= len(toks) <= 45:
                continue
            alpha = sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z'\-,.;:()]*", t)) for t in toks)
            if alpha / len(toks) < 0.7 or any(b in s.lower() for b in self.BOILER):
                continue
            out.append(s)
        return out

    def _id(self, w: str, create: bool = True) -> int:
        if w in self.w2i:
            return self.w2i[w]
        if not create or len(self.w2i) >= 30000:
            return 1
        i = len(self.w2i)
        self.w2i[w], self.i2w[i] = i, w
        return i

    def surprisal(self, words: List[str]) -> float:
        if len(words) < 2:
            return 0.0
        V, eps, tot = max(len(self.w2i) + 1, 2000), 0.5, 0.0     # floor on V: an empty brain is maximally surprised
        for w1, w2 in zip(words, words[1:]):
            i1, i2 = self.w2i.get(w1, -1), self.w2i.get(w2, -1)
            row = self.bi.get(i1, {})
            p = (row.get(i2, 0.0) + eps / V) / (self.bi_tot.get(i1, 0.0) + eps)
            tot += -math.log2(p)
        return tot / (len(words) - 1)

    def ingest(self, sentence: str, strength: float = 1.0) -> int:
        words = self.words_of(sentence)
        if len(words) < 2:
            return 0
        new = sum(1 for w in words if w not in self.w2i)
        ids = [self._id(w) for w in words]
        for i in ids:
            self.count[i] = self.count.get(i, 0) + 1
        for a, b in zip(ids, ids[1:]):
            self.bi.setdefault(a, {})
            self.bi[a][b] = self.bi[a].get(b, 0.0) + self.eta * strength
            self.bi_tot[a] = self.bi_tot.get(a, 0.0) + self.eta * strength
        for a, b, c in zip(ids, ids[1:], ids[2:]):
            row = self.tri.setdefault((a, b), {})
            row[c] = row.get(c, 0.0) + self.eta * strength * 2.5
        for p, a in enumerate(ids):
            row = self.co.setdefault(a, {})
            for q in range(max(0, p - 4), min(len(ids), p + 5)):
                if q != p:
                    row[ids[q]] = row.get(ids[q], 0.0) + self.eta * 0.4 * strength
        return new

    def decay(self, f: float = 0.97, floor: float = 0.05):
        """Homeostatic forgetting during sleep: weak/unused associations fade, memory stays bounded."""
        for tbl in (self.bi, self.co):
            for s in list(tbl):
                row = tbl[s]
                for d in list(row):
                    row[d] *= f
                    if row[d] < floor:
                        del row[d]
                if not row:
                    del tbl[s]
        for s in list(self.tri):
            row = self.tri[s]
            for d in list(row):
                row[d] *= f
                if row[d] < floor:
                    del row[d]
            if not row:
                del self.tri[s]
        self.bi_tot = {s: sum(r.values()) for s, r in self.bi.items()}

    def concepts(self, words: List[str]) -> List[str]:
        return [w for w in words if w not in self.FUNC and w in self.w2i]

    def _spread(self, topic: set) -> Dict[int, float]:
        act: Dict[int, float] = {}
        for t in topic:
            for n, v in self.co.get(t, {}).items():
                if self.i2w.get(n) not in self.FUNC:
                    act[n] = act.get(n, 0.0) + v
        tot = sum(act.values()) + 1e-6
        return {k: v / tot for k, v in act.items()}

    def generate(self, seeds: List[str], n: int = 12, temp: float = 0.6) -> str:
        ids = [self.w2i[w] for w in seeds if w in self.w2i]
        if not ids:
            return " ".join(seeds)
        out = list(ids)
        topic = {i for i in ids if self.i2w[i] not in self.FUNC}
        spread = self._spread(topic)
        for _ in range(n):
            cur, prev = out[-1], (out[-2] if len(out) > 1 else None)
            cand: Dict[int, float] = {}
            row = self.tri.get((prev, cur)) if prev is not None else None
            if row:
                tot = sum(row.values()) + 1e-6
                for k, v in row.items():
                    cand[k] = cand.get(k, 0.0) + 4.0 * v / tot
            row = self.bi.get(cur)
            if row:
                tot = self.bi_tot.get(cur, 0.0) + 1e-6
                scale = 1.0 if cand else 2.5
                for k, v in row.items():
                    cand[k] = cand.get(k, 0.0) + scale * v / tot
            if not cand:
                break
            for k in cand:
                cand[k] *= (1.0 + 30.0 * spread.get(k, 0.0)) / (1.0 + 3.0 * out.count(k))
            top = sorted(cand.items(), key=lambda kv: -kv[1])[:12]
            w = np.array([v for _, v in top], np.float64) ** (1.0 / temp)
            nxt = top[int(np.random.choice(len(top), p=w / w.sum()))][0]
            out.append(nxt)
            if self.i2w[nxt] not in self.FUNC:
                topic.add(nxt)
                spread = self._spread(topic)
        return " ".join(self.i2w[i] for i in out)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        words = list(self.w2i)
        flat = lambda tbl: [(s, d, v) for s, r in tbl.items() for d, v in r.items()]
        bi, co = flat(self.bi), flat(self.co)
        tri = [(a, b, d, v) for (a, b), r in self.tri.items() for d, v in r.items()]
        arr = lambda rows, c, dt: np.array([r[c] for r in rows], dtype=dt)
        tmp = path + ".tmp.npz"
        np.savez_compressed(
            tmp, words=np.array(words), ids=np.array([self.w2i[w] for w in words], np.int32),
            counts=np.array([self.count.get(self.w2i[w], 0) for w in words], np.int32),
            bi_src=arr(bi, 0, np.int32), bi_dst=arr(bi, 1, np.int32), bi_val=arr(bi, 2, np.float32),
            tri_s1=arr(tri, 0, np.int32), tri_s2=arr(tri, 1, np.int32), tri_dst=arr(tri, 2, np.int32),
            tri_val=arr(tri, 3, np.float32),
            co_src=arr(co, 0, np.int32), co_dst=arr(co, 1, np.int32), co_val=arr(co, 2, np.float32))
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        d = np.load(path, allow_pickle=True)
        self.w2i = {str(w): int(i) for w, i in zip(d["words"], d["ids"])}
        self.i2w = {i: w for w, i in self.w2i.items()}
        self.count = {int(i): int(c) for i, c in zip(d["ids"], d["counts"])}
        self.bi, self.tri, self.co = {}, {}, {}
        for s, t, v in zip(d["bi_src"], d["bi_dst"], d["bi_val"]):
            self.bi.setdefault(int(s), {})[int(t)] = float(v)
        for a, b, c, v in zip(d["tri_s1"], d["tri_s2"], d["tri_dst"], d["tri_val"]):
            self.tri.setdefault((int(a), int(b)), {})[int(c)] = float(v)
        if "co_src" in d:
            for s, t, v in zip(d["co_src"], d["co_dst"], d["co_val"]):
                self.co.setdefault(int(s), {})[int(t)] = float(v)
        self.bi_tot = {s: sum(r.values()) for s, r in self.bi.items()}
        return True


# =================================================================================================
# AWARENESS MACHINERY: DRIVES, GLOBAL WORKSPACE, SELF-MODEL
# =================================================================================================
class Drives:
    def __init__(self, sleep_every: int = 400):
        self.boredom, self.fatigue, self.social = 0.5, 0.0, 0.0
        self.sleep_every = sleep_every

    @property
    def curiosity(self) -> float:
        return float(np.clip(0.15 + 0.85 * self.boredom, 0.15, 1.0))

    def update(self, novelty: float, human: bool):
        self.boredom += 0.03 * ((1.0 - min(1.0, novelty * 2.0)) - self.boredom)
        self.fatigue += 1.0 / self.sleep_every
        self.social = 1.0 if human else self.social * 0.92

    def exploration(self, confidence: float) -> Tuple[float, float]:
        c = self.curiosity
        return 0.04 + 0.25 * c * (1.0 - 0.5 * confidence), 0.12 + 0.25 * c

    def rest(self):
        self.fatigue = 0.0


class GlobalWorkspace:
    """Tiny capacity-limited stage: contents compete by salience; the winner is broadcast to speech/memory."""

    def __init__(self, capacity: int = 4):
        self.items: List[Dict[str, Any]] = []
        self.capacity = capacity

    def post(self, src: str, words: List[str], sal: float):
        self.items.append({"src": src, "words": list(words), "sal": float(sal)})

    def tick(self):
        for it in self.items:
            it["sal"] *= 0.85
        self.items = sorted((i for i in self.items if i["sal"] > 0.05), key=lambda i: -i["sal"])[:self.capacity]

    def winner(self) -> Optional[Dict[str, Any]]:
        return self.items[0] if self.items else None


class SelfModel:
    """Running statistics about its own behaviour; confidence feeds back into exploration."""

    def __init__(self):
        self.ema_r, self.ema_nov, self.ema_err = 0.0, 0.5, 0.15
        self.counts = np.ones(N_ACTIONS)

    def observe(self, a: int, r: float, nov: float, err: float):
        self.ema_r += 0.02 * (r - self.ema_r)
        self.ema_nov += 0.02 * (nov - self.ema_nov)
        self.ema_err += 0.02 * (err - self.ema_err)
        self.counts[a] += 1

    @property
    def confidence(self) -> float:
        return 1.0 / (1.0 + 10.0 * self.ema_err)

    def report(self) -> Dict[str, float]:
        p = self.counts / self.counts.sum()
        return {"avg_reward": round(self.ema_r, 3), "novelty_rate": round(self.ema_nov, 3),
                "confidence": round(self.confidence, 3),
                "behaviour_entropy": round(float(-(p * np.log(p)).sum() / math.log(N_ACTIONS)), 3)}


# =================================================================================================
# THE ORGANISM (CORE MODEL)
# =================================================================================================
class Organism:
    """
    Unified cognitive agent integrating visual perception, sparse coding, episodic memory,
    forward prediction, TD-Hebbian reinforcement learning, homeostatic drives, and language cortex.
    """

    def __init__(self, env: Any = None, ckpt_dir: str = CKPT_DIR, seed: int = 0, sleep_every: int = 400,
                 explore_only: bool = False, verbose: bool = True, autosave_every: int = 200,
                 legacy_language: Optional[str] = None):
        self.env, self.verbose, self.explore_only = env, verbose, explore_only
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.ckpt = os.path.join(ckpt_dir, "organism_v2.npz")
        self.lang_path = os.path.join(ckpt_dir, "organism_v2_language.npz")
        self.autosave_every = autosave_every
        self.retina = Retina()
        self.encoder = SparseEncoder()
        self.mem = EpisodicMemory()
        self.fwd = ForwardModel()
        self.policy = TDHebbianPolicy(self.encoder.n_units, self.encoder.k)
        self.lang = LanguageCortex()
        self.drives, self.ws, self.me = Drives(sleep_every), GlobalWorkspace(), SelfModel()
        self.replay: deque = deque(maxlen=3000)
        self.dead: Dict[int, set] = {}
        self.prev: Optional[Dict[str, Any]] = None
        self.t, self.last_read_t, self.sleeps = 0, -99, 0
        self.thought, self.last_concepts = "", []
        self.pending_fb, self.human_msgs = 0.0, queue.Queue()
        self.replies: "queue.Queue[str]" = queue.Queue()
        self.running = False
        self.stats = {"clicks": 0, "effective": 0, "reward_sum": 0.0, "novel_arrivals": 0}
        self.load(legacy_language)

    # ---- human interface (thread-safe)
    def hear(self, text: str):
        self.human_msgs.put(text)

    def feedback(self, val: float):
        self.pending_fb += val

    # ---- perception-action step
    def step(self, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes one cognitive cycle.
        If obs is provided, uses it directly; otherwise pulls observation from self.env.
        obs format: {"rgb": np.ndarray (H, W, 3), "motion": np.ndarray (H, W), Optional["text"]: str}
        """
        self.t += 1
        if obs is None:
            if self.env is None:
                raise ValueError("No observation provided and no environment configured.")
            obs = self.env.observe()

        m6 = self.retina.encode(obs["rgb"], obs["motion"])
        x = self.retina.pooled(m6, obs["rgb"])
        F = self.retina.cell_features(m6, obs["rgb"])
        gf = F.mean(axis=0)
        idx = self.encoder.encode(x, learn=True)
        g = self.fwd.gist(x)
        ctx, nov, sim = self.mem.observe(x, self.t)

        human = self._digest_human()
        raw_text = obs.get("text") if "text" in obs else (self.env.poll_text() if self.env and hasattr(self.env, "poll_text") else None)
        r_text = self._digest_text(raw_text)
        fb, self.pending_fb = self.pending_fb, 0.0
        r, err, change, wasted = 0.0, self.me.ema_err, 0.0, False

        if self.prev is not None:
            p = self.prev
            err = self.fwd.update(p["g"], p["a"], g)
            if p["a"] != A_READ:
                change = float(np.clip((1.0 - float(x @ p["x"]) - 0.05) / 0.5, 0.0, 1.0))
                wasted = float(x @ p["x"]) > 0.985
            habit = 1.0 / (1.0 + 0.4 * float(self.mem.count[ctx]))       # sense-of-agency habituates on over-visited scenes
            r = (0.55 * nov + 0.15 * change * habit + 0.40 * r_text
                 + 0.25 * float(np.clip(self.fwd.progress * 5.0, -0.5, 0.5)) - (0.08 if wasted else 0.0) + fb)
            r = float(np.clip(r, -1.5, 1.5))
            st = (idx, F, gf)
            self.policy.update(p["state"], p["a"], r, st)
            self.mem.credit(p["ctx"], p["a"], r)
            self.replay.append((p["state"], p["a"], r, st))
            self.me.observe(p["a"], r, nov, err)
            self.stats["reward_sum"] += r
            if p["a"] < N_CELLS:
                self.stats["clicks"] += 1
                self.stats["effective"] += int(change > 0.3)
                self.stats["novel_arrivals"] += int(change > 0.3 and nov > 0.5)
            if wasted and p["a"] < N_CELLS:
                self.dead.setdefault(p["ctx"], set()).add(p["a"])      # inhibition of return, per scene

        self.drives.update(nov, human)
        self.ws.tick()
        if nov > 0.3:
            self.ws.post("novelty", self.last_concepts[:4], nov)

        if nov > 0.35 and self.t - self.last_read_t > 12:                 # novelty captures attention -> read
            if self.env and hasattr(self.env, "request_read"):
                self.env.request_read()
            self.last_read_t = self.t

        q = self.policy.q(idx, F, gf) + 0.5 * self.mem.Qe[ctx]
        for d in self.dead.get(ctx, ()):
            q[d] -= 0.3
        supports_typing = getattr(self.env, "supports_typing", False) if self.env else True
        if not supports_typing:
            q[A_TYPE] = -1e9

        eps, tau = self.drives.exploration(self.me.confidence)
        a = self.policy.select(q, 1.0 if self.explore_only else eps, tau, self.rng)
        text = self._pick_text() if a == A_TYPE else None

        info = self.env.act(a, text=text) if self.env and hasattr(self.env, "act") else action_name(a)

        if self.t % 6 == 0 or human or r_text > 0:
            w = self.ws.winner()
            seeds = (w["words"] if w and w["words"] else self.last_concepts)[:3]
            self.thought = self.lang.generate(seeds, n=10) if seeds else ""

        self.prev = {"state": (idx, F, gf), "a": a, "g": g, "x": x, "ctx": ctx}
        if self.autosave_every and self.t % self.autosave_every == 0:
            self.save()

        return {"t": self.t, "a": a, "info": info, "r": r, "nov": nov, "change": change, "pages": self.mem.n,
                "vocab": len(self.lang.w2i), "curiosity": self.drives.curiosity, "fatigue": self.drives.fatigue,
                "conf": self.me.confidence, "delta": self.policy.last_delta, "thought": self.thought}

    def _digest_text(self, txt: Optional[str]) -> float:
        if not txt:
            return 0.0
        new_words, best = 0, None
        for s in self.lang.sentences(txt):
            words = self.lang.words_of(s)
            sur = self.lang.surprisal(words)
            if sur < 2.0:                                                  # already predictable -> habituate
                continue
            new_words += self.lang.ingest(s, strength=1.0 + min(4.0, sur / 2.0))
            if best is None or sur > best[0]:
                best = (sur, words)
        if best:
            self.last_concepts = self.lang.concepts(best[1])[:6]
            self.ws.post("text", self.last_concepts, min(1.0, best[0] / 10.0))
        return min(1.0, new_words / 10.0)

    def _digest_human(self) -> bool:
        got = False
        while True:
            try:
                msg = self.human_msgs.get_nowait()
            except queue.Empty:
                return got
            got = True
            self.lang.ingest(msg, strength=4.0)
            words = self.lang.concepts(self.lang.words_of(msg))
            self.ws.post("human", words[:4], 1.0)
            self.replies.put(self.lang.generate(words[:3] or self.lang.words_of(msg)[:3], n=14, temp=0.5))

    def _pick_text(self) -> Optional[str]:
        w = self.ws.winner()
        words = (w["words"] if w else []) or self.last_concepts
        return random.choice(words) if words else None

    # ---- sleep: replay, forgetting, consolidation, checkpoint
    def sleep(self):
        self.sleeps += 1
        n = min(len(self.replay), 400)
        if n:
            for i in self.rng.choice(len(self.replay), n, replace=False):
                st, a, r, st2 = self.replay[int(i)]
                self.policy.replay_update(st, a, r, st2)
        self.lang.decay(0.97, 0.05)
        merged = self.mem.consolidate()
        self.dead.clear()
        self.prev = None
        self.drives.rest()
        self.save()
        if self.verbose:
            print(f"   💤 sleep #{self.sleeps}: replayed {n} episodes, merged {merged} scenes, "
                  f"vocab={len(self.lang.w2i)}, scenes={self.mem.n}, self={self.me.report()}")

    # ---- persistence
    def save(self):
        os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)
        tmp = self.ckpt + ".tmp.npz"
        np.savez_compressed(tmp, encW=self.encoder.W, encB=self.encoder.bias, encR=self.encoder.rate,
                            polW=self.policy.W, polV=self.policy.V, polU=self.policy.U, fwdM=self.fwd.M,
                            memP=self.mem.P, memC=self.mem.count, memL=self.mem.last, memQ=self.mem.Qe,
                            memN=np.array([self.mem.n]), retMu=self.retina.mu, retVar=self.retina.var,
                            retN=np.array([self.retina.n]), t=np.array([self.t]))
        os.replace(tmp, self.ckpt)
        self.lang.save(self.lang_path)

    def load(self, legacy_language: Optional[str] = None):
        if os.path.exists(self.ckpt):
            try:
                d = np.load(self.ckpt)
                if d["encW"].shape == self.encoder.W.shape and d["polW"].shape == self.policy.W.shape:
                    self.encoder.W, self.encoder.bias, self.encoder.rate = d["encW"], d["encB"], d["encR"]
                    self.policy.W, self.policy.V, self.policy.U = d["polW"], d["polV"], d["polU"]
                    self.fwd.M = d["fwdM"]
                    self.mem.P, self.mem.count, self.mem.last, self.mem.Qe = d["memP"], d["memC"], d["memL"], d["memQ"]
                    self.mem.n = int(d["memN"][0])
                    self.retina.mu, self.retina.var, self.retina.n = d["retMu"], d["retVar"], int(d["retN"][0])
                    self.t = int(d["t"][0])
            except Exception as e:
                if self.verbose:
                    print(f"   (checkpoint not loaded: {e})")
        if not self.lang.load(self.lang_path) and legacy_language:
            if self.lang.load(legacy_language):
                if self.verbose:
                    print(f"   (imported language memory from {legacy_language})")

    # ---- run loop
    def run(self, max_steps: int = 0, delay: float = 0.0):
        self.running, errors = True, 0
        while self.running:
            t_c = time.time()
            try:
                rec = self.step()
                errors = 0
                if self.verbose:
                    print(self.format(rec))
                while not self.replies.empty():
                    print(f"   🗣️  {self.replies.get_nowait()}")
                if self.drives.fatigue >= 1.0:
                    self.sleep()
            except Exception as e:
                errors += 1
                print(f"   ⚠ step error ({errors}): {e!r}")
                if errors >= 5:
                    break
            if max_steps and self.t >= max_steps:
                break
            if delay > 0:
                time.sleep(max(0.0, delay - (time.time() - t_c)))
        self.running = False
        self.save()

    @staticmethod
    def format(rec: Dict[str, Any]) -> str:
        th = f' 💭 "{rec["thought"]}"' if rec["thought"] else ""
        return (f"[{rec['t']:05d}] {action_name(rec['a']):<13} {str(rec['info'])[:18]:<18} r={rec['r']:+.2f} "
                f"nov={rec['nov']:.2f} chg={rec['change']:.2f} | scenes={rec['pages']:3d} words={rec['vocab']:4d} | "
                f"cur={rec['curiosity']:.2f} fat={rec['fatigue']:.2f} conf={rec['conf']:.2f} TD={rec['delta']:+.2f}{th}")


# =================================================================================================
# SYNTHETIC TEST ENVIRONMENT & VERIFICATION
# =================================================================================================
_MOCK_TEXT = [
    "the visual cortex organizes retinal signals into oriented edge detectors",
    "volcanic islands form where tectonic plates slowly spread apart",
    "medieval merchants carried spices along caravan routes across the desert",
    "honeybees communicate the location of flowers through a waggle dance",
    "ocean currents redistribute heat between the equator and the poles",
    "early printing presses made books affordable for ordinary readers",
    "bridges distribute load through tension compression and careful geometry",
    "fermentation lets yeast convert sugar into alcohol and carbon dioxide",
    "migrating birds navigate using magnetic fields and the position of stars",
    "glaciers carve deep valleys while carrying enormous quantities of sediment",
    "ancient astronomers tracked planetary motion with patient nightly observation",
    "coral reefs shelter a remarkable diversity of fish and invertebrates",
    "electric grids balance supply and demand every second of the day",
    "monsoon winds reverse direction with the seasons across southern asia",
    "mycelium networks exchange nutrients between neighbouring forest trees",
    "the printing of maps advanced when engravers learned to share copper plates",
]


class MockWeb:
    """
    Synthetic multi-page sensory environment for testing cognitive learning dynamics.
    Pages feature distinct textures, blue interactive link regions, and natural text.
    """

    def __init__(self, n_pages: int = 30, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.n_pages = n_pages
        self.pages = []
        for p in range(n_pages):
            tex = np.kron(self.rng.rand(9, 16, 3), np.ones((4, 4, 1)))
            stripes = (self.rng.rand(GRID_H, 1) > 0.6).astype(np.float32)
            base = self.rng.rand(3) * 0.7 + 0.1
            links = list(self.rng.choice(N_CELLS, 3, replace=False))
            targets = [int(self.rng.choice([q for q in range(n_pages) if q != p])) for _ in links]
            self.pages.append({"tex": tex, "stripes": stripes, "base": base, "links": links,
                               "targets": targets, "text": _MOCK_TEXT[p % len(_MOCK_TEXT)] + f" number {p}."})
        self.cur, self.hist, self.scroll, self.visited = 0, [], 0, {0}
        self.prev_lum: Optional[np.ndarray] = None
        self._pending: Optional[str] = None
        self.supports_typing = False
        self.clicks = self.effective = 0

    def _render(self) -> np.ndarray:
        pg = self.pages[self.cur]
        img = pg["base"][None, None, :] + 0.15 * (pg["tex"] - 0.5)
        img = img + 0.25 * np.roll(pg["stripes"], 5 * self.scroll, axis=0)[:, :, None] * np.ones(3)
        for c in pg["links"]:
            col, row = c % CELL_COLS, c // CELL_COLS
            y0, x0 = row * (GRID_H // CELL_ROWS), col * (GRID_W // CELL_COLS)
            img[y0:y0 + 9, x0:x0 + 8] = (0.10, 0.20, 0.85)
            img[y0 + 2:y0 + 9:3, x0 + 1:x0 + 7] = (0.45, 0.55, 0.95)
        return np.clip(img + self.rng.randn(GRID_H, GRID_W, 3) * 0.01, 0, 1).astype(np.float32)

    def observe(self) -> Dict[str, Any]:
        rgb = self._render()
        lum = Retina.lum_of(rgb)
        motion = np.abs(lum - self.prev_lum) if self.prev_lum is not None else np.zeros_like(lum)
        self.prev_lum = lum
        return {"rgb": rgb, "motion": motion.astype(np.float32)}

    def request_read(self):
        self._pending = self.pages[self.cur]["text"]

    def poll_text(self) -> Optional[str]:
        t, self._pending = self._pending, None
        return t

    def act(self, a: int, text: Optional[str] = None) -> str:
        pg = self.pages[self.cur]
        if a < N_CELLS:
            self.clicks += 1
            if a in pg["links"] and self.rng.rand() > 0.05:
                self.hist.append(self.cur)
                self.cur = pg["targets"][pg["links"].index(a)]
                self.scroll = 0
                self.visited.add(self.cur)
                self.effective += 1
                return f"->page{self.cur}"
        elif a == A_SCROLL_DOWN:
            self.scroll = min(2, self.scroll + 1)
        elif a == A_SCROLL_UP:
            self.scroll = max(0, self.scroll - 1)
        elif a == A_BACK and self.hist:
            self.cur = self.hist.pop()
            self.scroll = 0
        elif a == A_READ:
            self.request_read()
        return "noop"


def _run_mock(seed: int, steps: int, explore_only: bool) -> Dict[str, Any]:
    env = MockWeb(n_pages=30, seed=seed)
    org = Organism(env, ckpt_dir=tempfile.mkdtemp(), seed=seed, explore_only=explore_only,
                   verbose=False, autosave_every=0, sleep_every=300)
    half = steps // 2
    c_eff = [0, 0]
    reached_at = None
    for i in range(steps):
        org.step()
        if org.drives.fatigue >= 1.0:
            org.sleep()
        if i == half:
            c_eff = [env.clicks, env.effective]
        if reached_at is None and len(env.visited) >= 20:
            reached_at = i
    late_clicks = max(1, env.clicks - c_eff[0])
    return {"visited": len(env.visited), "late_effective_rate": (env.effective - c_eff[1]) / late_clicks,
            "reached20": reached_at if reached_at is not None else steps, "vocab": len(org.lang.w2i),
            "scenes": org.mem.n, "org": org}


def selftest() -> bool:
    """Verifies the core cognitive model dynamics: learning, habituation, language, memory, sleep."""
    ok_all = True

    def check(name: str, ok: bool, extra: str = ""):
        nonlocal ok_all
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")

    print("== Learning vs. Random Baseline on Synthetic Web (5 seeds x 1500 steps) ==")
    L, R = [], []
    for s in range(5):
        L.append(_run_mock(s, 1500, False))
        R.append(_run_mock(s, 1500, True))
    m = lambda rs, k: float(np.mean([r[k] for r in rs]))
    print(f"  {'':22}{'learner':>10}{'random':>10}")
    for k, lab in (("late_effective_rate", "effective-click rate (2nd half)"), ("visited", "pages visited (of 30)"),
                   ("reached20", "steps to visit 20 pages")):
        print(f"  {lab:<32}{m(L, k):>10.3f}{m(R, k):>10.3f}")
    check("learner clicks links far more often than chance",
          m(L, "late_effective_rate") > 2.0 * m(R, "late_effective_rate"),
          f"({m(L, 'late_effective_rate'):.2f} vs {m(R, 'late_effective_rate'):.2f})")
    check("learner covers the web faster (mean steps to 20 pages)",
          m(L, "reached20") < 0.6 * m(R, "reached20"),
          f"({m(L, 'reached20'):.0f} vs {m(R, 'reached20'):.0f})")

    print("\n== Language / Sleep / Persistence ==")
    org = L[0]["org"]
    check("learned vocabulary from sensory text", len(org.lang.w2i) > 40, f"(vocab={len(org.lang.w2i)})")
    s_before = org.lang.surprisal(org.lang.words_of(_MOCK_TEXT[0]))
    s_novel = org.lang.surprisal(org.lang.words_of("quantum chromodynamics describes gluon exchange"))
    check("habituation: seen text less surprising than unseen text",
          s_before < s_novel, f"({s_before:.1f} < {s_novel:.1f} bits)")
    thought = org.lang.generate(["cortex"], n=8)
    check("inner speech generates from learned associations", len(thought.split()) > 1, f'("{thought}")')
    org.save()
    org2 = Organism(MockWeb(seed=0), ckpt_dir=os.path.dirname(org.ckpt), verbose=False, autosave_every=0)
    check("checkpoint round-trip", org2.mem.n == org.mem.n and len(org2.lang.w2i) == len(org.lang.w2i))
    n_before = len(org.lang.bi)
    org.sleep()
    check("sleep consolidation (replay + decay)", org.sleeps >= 1 and len(org.lang.bi) <= n_before)
    print("\nSELF-TEST", "PASSED" if ok_all else "FAILED")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description="Organism: Core Cognitive Model (zero backprop)")
    ap.add_argument("--selftest", action="store_true", help="run comprehensive verification suite")
    ap.add_argument("--steps", type=int, default=100, help="run mock learning steps (default: 100)")
    ap.add_argument("--sleep-every", type=int, default=300)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    print("Running Organism core model in synthetic environment...")
    env = MockWeb(n_pages=30, seed=42)
    org = Organism(env, sleep_every=args.sleep_every)
    org.run(max_steps=args.steps)
    print(f"\nDone. Steps={org.t}, Scenes={org.mem.n}, Vocab={len(org.lang.w2i)}, Stats={org.me.report()}")


if __name__ == "__main__":
    main()