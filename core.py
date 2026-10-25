#!/usr/bin/env python3
"""
===================================================================================================
ORGANISM: CORE COGNITIVE MODEL (NumPy + Pillow only, strictly zero backpropagation)
===================================================================================================
A biologically grounded perception -> memory -> motivation -> action cognitive architecture that
learns continuously from multi-modal sensory experience (vision, text, movement, feedback) without
backpropagation, loss gradients, autograd, or external deep learning frameworks.

ARCHITECTURAL PRINCIPLES:
  1. Biologically Plausible Local Plasticity:
     - Synaptic modification is strictly local in space and time (Oja subspace projection,
       Hebbian association, and three-factor eligibility trace reinforcement).
  2. Sparse Competitive Coding (k-WTA):
     - Cortical representations are formed through competitive k-Winners-Take-All dynamics with
       homeostatic inhibitory thresholds to prevent runaway neural saturation.
  3. Non-Parametric Episodic Engram Memory:
     - Perceptual exemplars are stored as unit-normalized prototypes with novelty quantification,
       one-shot credit assignment, and offline sleep-time consolidation (prototype merging).
  4. Predictive Sensorimotor Forward Model:
     - The agent continuously anticipates the perceptual consequences of its motor actions
       (predictive processing). The mismatch between predicted and actual sensory state yields
       epistemic surprise and drives intrinsic motivation (artificial curiosity).
  5. Multi-Factor Reinforcement Learning without Gradients:
     - Policy learning uses TD(lambda) error as a neuromodulatory third factor that gates
       Hebbian eligibility traces across both cortical sparse codes and visual affordance fields.
  6. Associative Language Cortex:
     - Lexical, bigram, trigram, and semantic co-occurrence networks are acquired directly from
       sensory text, modulated by information-theoretic surprisal, with inner monologue generation.
  7. Homeostatic Motivational Drives:
     - Internal drives (boredom, curiosity, fatigue, and social engagement) dynamically balance
     - exploration vs. exploitation and regulate circadian sleep/wake cycles.
  8. Global Workspace Working Memory:
     - A capacity-limited competitive stage where salient sensory, motivational, and semantic
       representations compete; winning contents are broadcast across the cognitive architecture.
  9. Metacognitive Self-Monitoring:
     - Real-time tracking of subjective confidence, prediction error, and behavioral entropy.

SYSTEM DATAFLOW PER TIMESTEP:
  [Raw RGB & Motion] ---> [Retina Encoder] ---> (Pooled 870D Scene Vector x, Local Features F)
                                 |
                                 v
                         [SparseEncoder] ---> Cortical Active Indices (k=24)
                                 |
                                 +---> [EpisodicMemory] ---> Prototype ctx, Novelty, Similarity
                                 |
                                 +---> [ForwardModel]   ---> Surprise error & Learning Progress
                                 |
                                 +---> [GlobalWorkspace] <--- Language Surprisal & Social Signals
                                 |             |
                                 v             v
                       [TDHebbianPolicy] <--- [Drives] (Curiosity, Temperature tau, Epsilon eps)
                                 |
                                 v
                          [Motor Action] ---> Executed in Environment ---> Next Perception
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
# GEOMETRY / ACTION SPACE / SYSTEM CONSTANTS
# =================================================================================================

# Visual Sensory Geometry
GRID_W: int = 64
"""Retinal visual field horizontal resolution in pixels."""

GRID_H: int = 36
"""Retinal visual field vertical resolution in pixels."""

# Fixation Receptive Field Grid
CELL_COLS: int = 8
"""Number of horizontal fixation target columns across the content area."""

CELL_ROWS: int = 4
"""Number of vertical fixation target rows across the content area."""

N_CELLS: int = CELL_COLS * CELL_ROWS
"""Total number of spatial click/fixation cells (8 x 4 = 32 discrete spatial targets)."""

# Motor Action Space Indices
A_SCROLL_DOWN: int = N_CELLS
"""Action index (32): Scroll down the content viewport."""

A_SCROLL_UP: int = N_CELLS + 1
"""Action index (33): Scroll up the content viewport."""

A_READ: int = N_CELLS + 2
"""Action index (34): Dwell gaze on screen to parse and comprehend readable text."""

A_TYPE: int = N_CELLS + 3
"""Action index (35): Select and type a learned concept word into active input field."""

A_BACK: int = N_CELLS + 4
"""Action index (36): Execute navigation backstep to previous viewpoint."""

N_ACTIONS: int = N_CELLS + 5
"""Total discrete motor action space cardinality (32 spatial targets + 5 global primitives = 37)."""

# Feature Representations Dimensionality
N_LOC: int = 8
"""Number of local sensory features extracted per fixation cell (6 base channels + variance + color)."""

X_DIM: int = 6 * (GRID_H // 4) * (GRID_W // 4) + 6
"""
Dimensionality of the pooled holistic scene vector:
  - 6 retinal channels spatially downsampled by 4x4 block pooling:
    (36 // 4) * (64 // 4) = 9 * 16 = 144 blocks per channel.
  - 6 channels * 144 blocks = 864 features.
  - 6 global mean color/channel intensities appended.
  - Total dimensionality = 864 + 6 = 870 dimensions.
"""

# Filesystem Checkpoint Directory
CKPT_DIR: str = os.path.abspath("checkpoints")
"""Default directory path for persisting neural weight matrices and cognitive memories."""


def action_name(a: int) -> str:
    """
    Translates an integer action index into a human-readable descriptive string.

    Args:
        a (int): Discrete action identifier in range [0, N_ACTIONS - 1].

    Returns:
        str: Human-readable action label (e.g., 'click(col, row)', 'scroll_down', 'read', 'type', 'back').
    """
    if a < N_CELLS:
        col = a % CELL_COLS
        row = a // CELL_COLS
        return f"click({col},{row})"
    action_map = {
        A_SCROLL_DOWN: "scroll_down",
        A_SCROLL_UP: "scroll_up",
        A_READ: "read",
        A_TYPE: "type",
        A_BACK: "back"
    }
    return action_map.get(a, f"unknown_action({a})")


# =================================================================================================
# PERCEPTION: RETINA
# =================================================================================================
class Retina:
    """
    Biological receptive field visual encoder.

    Simulates the functional pathways of mammalian retina and early visual cortex (V1):
      1. Photopic Luminance: ITU-R BT.601 weighted luminance conversion.
      2. Color Opponency: Saturation and Blue-Yellow opponent signals.
      3. Motion Flux: Temporal frame difference between successive fixations.
      4. Oriented Spatial Gradients: Horizontal (gx) and vertical (gy) edge filters.
      5. Center-Surround Contrast: Difference between central pixel and 4-connected spatial neighbors.
      6. Spatial Pooling: Invariant 870D holistic scene vector for episodic memory and sparse coding.
      7. Receptive Field Affordances: 8D z-scored local feature vectors for each of the 32 click cells.

    Attributes:
        mu (np.ndarray): Running mean vector of local cell features across time (shape: (8,), float32).
        var (np.ndarray): Running variance vector of local cell features across time (shape: (8,), float32).
        n (int): Cumulative count of processed visual observations for online Welford statistics.
    """

    def __init__(self) -> None:
        """Initializes the retinal receptive field encoder and online running normalization statistics."""
        self.mu: np.ndarray = np.zeros(N_LOC, dtype=np.float32)
        self.var: np.ndarray = np.ones(N_LOC, dtype=np.float32)
        self.n: int = 0

    @staticmethod
    def lum_of(rgb: np.ndarray) -> np.ndarray:
        """
        Computes photopic luminance from an RGB array using the standard ITU-R BT.601 formula.

        Args:
            rgb (np.ndarray): Color image array of shape (..., 3) with channel values in [0.0, 1.0].

        Returns:
            np.ndarray: Grayscale luminance map of shape (...) matching the input spatial dimensions.
        """
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    def encode(self, rgb: np.ndarray, motion: np.ndarray) -> np.ndarray:
        """
        Transforms raw RGB pixels and motion flux into a 6-channel biological feature representation.

        The 6 feature channels correspond to:
          - Channel 0: Photopic Luminance (overall brightness).
          - Channel 1: Color Saturation (max(RGB) - min(RGB)).
          - Channel 2: Optical Motion Flux (clipped temporal change).
          - Channel 3: Horizontal Spatial Gradient (|d/dx| edge energy).
          - Channel 4: Vertical Spatial Gradient (|d/dy| edge energy).
          - Channel 5: Center-Surround Contrast (|pixel - local 4-neighbor average|).

        Args:
            rgb (np.ndarray): Visual RGB frame of shape (GRID_H, GRID_W, 3) in [0.0, 1.0].
            motion (np.ndarray): Optical motion/difference map of shape (GRID_H, GRID_W).

        Returns:
            np.ndarray: 6-channel feature map of shape (GRID_H, GRID_W, 6), float32.
        """
        # 1. Photopic Luminance
        lum = self.lum_of(rgb)

        # 2. Chromatic Saturation
        sat = rgb.max(-1) - rgb.min(-1)

        # 3. Spatial Derivatives / Oriented Edge Filters (horizontal and vertical)
        gx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
        gy = np.abs(np.diff(lum, axis=0, append=lum[-1:, :]))

        # 4. Center-Surround Neighborhood Contrast (Laplacian-like local contrast)
        nb = (
            np.roll(lum, 1, axis=0) +
            np.roll(lum, -1, axis=0) +
            np.roll(lum, 1, axis=1) +
            np.roll(lum, -1, axis=1)
        ) / 4.0
        contrast = np.abs(lum - nb)

        # Stack into 6-channel cortical feature tensor
        m6 = np.stack([
            lum,
            sat,
            np.clip(motion, 0.0, 1.0),
            gx,
            gy,
            contrast
        ], axis=-1)
        return m6.astype(np.float32)

    def pooled(self, m6: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """
        Constructs an invariant, layout-sensitive, unit-normalized holistic scene vector (870D).

        Translates spatial maps into block-averaged regional descriptors. Motion flux is deliberately
        omitted from this holistic identity signature, ensuring that transient motion does not
        artificially distort invariant scene identification in episodic memory.

        Channels included:
          - Luminance (m6 channel 0)
          - Saturation (m6 channel 1)
          - Horizontal Edge Gradient (m6 channel 3)
          - Vertical Edge Gradient (m6 channel 4)
          - Center-Surround Contrast (m6 channel 5)
          - Blue-Yellow Color Opponency: B - 0.5 * (R + G)

        Args:
            m6 (np.ndarray): 6-channel retinal feature map of shape (GRID_H, GRID_W, 6).
            rgb (np.ndarray): Original RGB frame of shape (GRID_H, GRID_W, 3).

        Returns:
            np.ndarray: Unit-normalized 870-dimensional scene vector x, float32.
        """
        # Blue-Yellow color opponency map
        by = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])

        # Selected 6 invariant feature maps (excluding transient motion channel 2)
        maps = [m6[..., 0], m6[..., 1], m6[..., 3], m6[..., 4], m6[..., 5], by]
        parts: List[np.ndarray] = []
        means: List[float] = []

        # Downsample each map into 4x4 pooling blocks (yielding 9x16 grid = 144 blocks per map)
        for mp in maps:
            # Reshape into (9 rows, 4 px, 16 cols, 4 px) and compute spatial mean over pooling blocks
            pooled_block = mp.reshape(GRID_H // 4, 4, GRID_W // 4, 4).mean(axis=(1, 3))
            means.append(float(pooled_block.mean()))
            # Standardize local variation and flatten to 144D
            std_val = float(pooled_block.std())
            normalized_block = ((pooled_block - pooled_block.mean()) / (std_val + 0.05)).ravel()
            parts.append(normalized_block)

        # Concatenate 6 * 144 = 864 block features + 6 global mean intensities (scaled by 3.0)
        v = np.concatenate(parts + [np.array(means, dtype=np.float32) * 3.0]).astype(np.float32)

        # Project to unit hypersphere for stable cosine similarity
        norm = float(np.linalg.norm(v))
        return v / (norm + 1e-8)

    def cell_features(self, m6: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """
        Extracts 8-dimensional local affordance feature vectors for each of the 32 click cells.

        For each of the 8x4 receptive grid cells:
          - Computes spatial mean across the cell's pixels for all 6 retinal channels.
          - Computes luminance standard deviation within the cell (texture complexity indicator).
          - Computes mean Blue-Yellow opponency within the cell (hyperlink/interactive cue).
          - Standardizes features online using running mean (mu) and variance (var).

        Args:
            m6 (np.ndarray): 6-channel retinal feature map of shape (GRID_H, GRID_W, 6).
            rgb (np.ndarray): Original RGB frame of shape (GRID_H, GRID_W, 3).

        Returns:
            np.ndarray: Normalized feature matrix F of shape (N_CELLS, N_LOC) = (32, 8), float32.
        """
        lum = m6[..., 0]
        by = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
        ch = GRID_H // CELL_ROWS  # 36 // 4 = 9 pixels per cell row
        cw = GRID_W // CELL_COLS  # 64 // 8 = 8 pixels per cell column

        # Mean across pixels within each of the 8x4 grid cells for all 6 channels -> (32, 6)
        base = m6.reshape(CELL_ROWS, ch, CELL_COLS, cw, 6).mean(axis=(1, 3)).reshape(N_CELLS, 6)

        # Luminance standard deviation within each cell -> (32, 1)
        lstd = lum.reshape(CELL_ROWS, ch, CELL_COLS, cw).std(axis=(1, 3)).reshape(N_CELLS, 1)

        # Mean Blue-Yellow opponency within each cell -> (32, 1)
        byc = by.reshape(CELL_ROWS, ch, CELL_COLS, cw).mean(axis=(1, 3)).reshape(N_CELLS, 1)

        # Concatenate into 8D raw feature vector per fixation cell
        raw = np.concatenate([base, lstd, byc], axis=1).astype(np.float32)

        # Online running statistics update (rapid initial adaptation, slower mature tracking)
        alpha = 0.05 if self.n < 50 else 0.005
        self.mu += alpha * (raw.mean(axis=0) - self.mu)
        self.var += alpha * (raw.var(axis=0) - self.var)
        self.n += 1

        # Z-score standardization with soft outlier clipping in [-3.0, 3.0]
        normalized = (raw - self.mu) / np.sqrt(self.var + 1e-4)
        return np.clip(normalized, -3.0, 3.0).astype(np.float32) * 0.5


# =================================================================================================
# CORTEX: SPARSE COMPETITIVE CODE, EPISODIC MEMORY, FORWARD MODEL
# =================================================================================================
class SparseEncoder:
    """
    Cortical sparse coding layer via k-Winners-Take-All (k-WTA) competitive Hebbian learning.

    Simulates cortical minicolumn sparse representation. Neurons compete via lateral inhibition;
    the top-k neurons with the highest membrane potential fire, while all other units are suppressed.
    Winning synaptic weights are updated via an unsupervised Hebbian attraction rule toward the input,
    followed by unit normalization (Oja-like sphere projection).

    To prevent "winner-take-all" monopolization by a few hyperactive units, a homeostatic inhibitory
    bias dynamically raises the firing threshold of frequently active neurons.

    Attributes:
        W (np.ndarray): Synaptic weight matrix of shape (n_units, n_in), unit-normalized rows.
        bias (np.ndarray): Homeostatic inhibitory threshold offsets (shape: (n_units,), float32).
        rate (np.ndarray): Exponential moving average of unit firing frequencies (shape: (n_units,), float32).
        n_units (int): Total population size of cortical units (default: 2048).
        k (int): Number of winning neurons permitted to fire per sensory saccade (default: 24).
        eta (float): Hebbian plasticity rate for synaptic weight modification (default: 0.05).
    """

    def __init__(self, n_in: int = X_DIM, n_units: int = 2048, k: int = 24, eta: float = 0.05, seed: int = 1) -> None:
        """
        Initializes the sparse cortical encoder with randomly oriented unit-normalized receptive fields.

        Args:
            n_in (int): Dimension of sensory input vector x (default: X_DIM = 870).
            n_units (int): Number of competitive cortical units (default: 2048).
            k (int): Number of active winners (sparsity level; default: 24).
            eta (float): Hebbian learning rate (default: 0.05).
            seed (int): Seed for deterministic random weight initialization.
        """
        rng = np.random.RandomState(seed)
        weights = rng.randn(n_units, n_in).astype(np.float32)
        # Normalize each unit's receptive field to unit hypersphere
        self.W: np.ndarray = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        self.bias: np.ndarray = np.zeros(n_units, dtype=np.float32)
        self.rate: np.ndarray = np.full(n_units, k / n_units, dtype=np.float32)
        self.n_units: int = n_units
        self.k: int = k
        self.eta: float = eta

    def encode(self, x: np.ndarray, learn: bool = True) -> np.ndarray:
        """
        Encodes an 870D sensory vector into a k-sparse binary index representation.

        Args:
            x (np.ndarray): Input sensory scene vector of shape (n_in,).
            learn (bool): If True, applies competitive Hebbian synaptic plasticity and homeostatic adjustments.

        Returns:
            np.ndarray: 1D array of length k containing the integer indices of the winning cortical units.
        """
        # Compute membrane potential: somatic dot product + homeostatic inhibitory bias
        potential = self.W @ x + self.bias

        # k-Winners-Take-All selection using partial partition for high computational efficiency
        winning_indices = np.argpartition(potential, -self.k)[-self.k:]

        if learn:
            # Unsupervised Hebbian plasticity: pull winning receptive fields toward the active stimulus
            w_winners = self.W[winning_indices]
            w_winners += self.eta * (x[None, :] - w_winners)
            # Re-normalize to unit hypersphere
            self.W[winning_indices] = w_winners / (np.linalg.norm(w_winners, axis=1, keepdims=True) + 1e-8)

            # Update exponential moving average of firing rates
            fired = np.zeros(self.n_units, dtype=np.float32)
            fired[winning_indices] = 1.0
            self.rate += 0.002 * (fired - self.rate)

            # Homeostatic inhibitory threshold adjustment:
            # Overactive neurons receive negative bias (harder to fire), underactive neurons get disinhibited
            target_rate = self.k / self.n_units
            self.bias = -1.5 * (self.rate - target_rate)

        return winning_indices


class EpisodicMemory:
    """
    Episodic prototype memory inspired by hippocampal engram circuits.

    Stores canonical scene exemplars (prototypes) on a unit hypersphere.
    Provides three primary cognitive functions:
      1. Novelty & Familiarity Scoring: Evaluates how distinct a new scene is from all past prototypes.
      2. Dynamic Prototype Allocation & Updating: Creates new memory slots for sufficiently novel scenes
         or refines existing prototypes via moving-average interpolation.
      3. One-Shot Contextual Credit Assignment: Caches episodic Q-values (Qe) per prototype to enable
         rapid tabular recall for specific familiar scenes.
      4. Sleep-Time Synaptic Consolidation: Merges near-duplicate prototypes during sleep to prune redundant
         memory slots and preserve capacity.

    Attributes:
        P (np.ndarray): Matrix of stored scene prototypes of shape (capacity, dim), float32.
        count (np.ndarray): Encounter tally / visit frequency for each prototype (shape: (capacity,), float32).
        last (np.ndarray): Timestep of the most recent encounter with each prototype (shape: (capacity,), float32).
        Qe (np.ndarray): Episodic Q-value lookup table caching expected action rewards per prototype (shape: (capacity, N_ACTIONS)).
        n (int): Current number of active prototypes stored in memory.
        cap (int): Maximum capacity limit of the prototype memory bank.
        alloc (float): Cosine distance allocation threshold (distance > alloc triggers allocation of a new prototype).
    """

    def __init__(self, dim: int = X_DIM, capacity: int = 600, alloc: float = 0.30) -> None:
        """
        Initializes the episodic prototype memory bank.

        Args:
            dim (int): Vector dimensionality of each prototype (default: X_DIM = 870).
            capacity (int): Maximum number of prototype engrams to retain (default: 600).
            alloc (float): Novelty threshold for allocating a new prototype slot (default: 0.30).
        """
        self.P: np.ndarray = np.zeros((capacity, dim), dtype=np.float32)
        self.count: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.last: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.Qe: np.ndarray = np.zeros((capacity, N_ACTIONS), dtype=np.float32)
        self.n: int = 0
        self.cap: int = capacity
        self.alloc: float = alloc

    def observe(self, x: np.ndarray, step: int) -> Tuple[int, float, float]:
        """
        Compares an incoming scene vector against stored prototypes.

        Computes cosine similarity, calculates novelty score, and either allocates a new prototype
        or adapts the nearest matching prototype.

        Args:
            x (np.ndarray): Unit-normalized scene vector of shape (dim,).
            step (int): Current global lifetime cognitive step index.

        Returns:
            Tuple[int, float, float]:
                - j (int): Index of the matching or newly allocated prototype slot.
                - novelty (float): Normalized novelty metric in [0.0, 1.0].
                - sim (float): Cosine similarity to the closest prototype in [-1.0, 1.0].
        """
        # First observation ever: allocate slot 0 immediately
        if self.n == 0:
            self._put(0, x, step)
            self.n = 1
            return 0, 1.0, 0.0

        # Compute cosine similarities between incoming scene x and all active prototypes
        similarities = self.P[:self.n] @ x
        best_idx = int(np.argmax(similarities))
        max_sim = float(similarities[best_idx])

        # Novelty score: calibrated linearly from similarity (sim < 0.40 -> novelty 1.0; sim > 0.90 -> novelty 0.0)
        novelty = float(np.clip((1.0 - max_sim - 0.10) / 0.5, 0.0, 1.0))

        # Check if difference exceeds the prototype allocation boundary
        cosine_distance = 1.0 - max_sim
        if cosine_distance > self.alloc:
            if self.n < self.cap:
                # Room available: allocate next available memory slot
                target_slot = self.n
                self.n += 1
            else:
                # Capacity reached: evict the least frequently and least recently visited prototype
                usage_metric = self.count[:self.n] + 1e-4 * self.last[:self.n]
                target_slot = int(np.argmin(usage_metric))
            self._put(target_slot, x, step)
            best_idx = target_slot
        else:
            # Familiar scene: attract nearest prototype coordinates slightly toward current input
            adapted = self.P[best_idx] + 0.1 * (x - max_sim * self.P[best_idx])
            self.P[best_idx] = adapted / (np.linalg.norm(adapted) + 1e-8)
            self.count[best_idx] += 1
            self.last[best_idx] = step

        return best_idx, novelty, max_sim

    def _put(self, j: int, x: np.ndarray, step: int) -> None:
        """
        Writes a new sensory prototype into slot j, resetting its count, timestamp, and Q-values.

        Args:
            j (int): Target memory slot index.
            x (np.ndarray): Unit-normalized scene vector.
            step (int): Global timestep of insertion.
        """
        self.P[j] = x
        self.count[j] = 1.0
        self.last[j] = float(step)
        self.Qe[j] = 0.0

    def credit(self, j: int, a: int, r: float) -> None:
        """
        Applies a local delta-rule reinforcement credit update to prototype j's episodic action value.

        Args:
            j (int): Prototype index to credit.
            a (int): Motor action executed in context j.
            r (float): Scalar reinforcement reward received.
        """
        self.Qe[j, a] += 0.3 * (r - self.Qe[j, a])

    def consolidate(self, thresh: float = 0.96) -> int:
        """
        Sleep-time synaptic consolidation: merges near-duplicate prototypes whose similarity exceeds thresh.

        Prunes redundant memory slots and aggregates encounter counts and episodic values.

        Args:
            thresh (float): Cosine similarity threshold for declaring two prototypes duplicates (default: 0.96).

        Returns:
            int: Number of duplicate prototype slots merged and removed.
        """
        n = self.n
        merged = 0
        dead_indices = set()

        # Compute full prototype-to-prototype cosine similarity matrix
        sim_matrix = self.P[:n] @ self.P[:n].T

        for i in range(n):
            if i in dead_indices:
                continue
            for j in range(i + 1, n):
                if j not in dead_indices and sim_matrix[i, j] > thresh:
                    # Keep the more frequently encountered prototype; merge into it
                    keep, drop = (i, j) if self.count[i] >= self.count[j] else (j, i)
                    self.count[keep] += self.count[drop]
                    self.Qe[keep] = 0.5 * (self.Qe[keep] + self.Qe[drop])
                    dead_indices.add(drop)
                    merged += 1
                    if drop == i:
                        break

        # Compact memory arrays by removing dead entries
        if dead_indices:
            surviving_indices = [idx for idx in range(n) if idx not in dead_indices]
            m = len(surviving_indices)
            for array in (self.P, self.count, self.last, self.Qe):
                array[:m] = array[surviving_indices]
            self.n = m

        return merged


class ForwardModel:
    """
    Sensorimotor predictive forward model.

    Implements the predictive processing paradigm: given current sensory gist g and planned motor
    action a, predicts the next sensory gist g_next. Discrepancy between prediction and reality
    yields epistemic surprise. Tracking the rate of change of prediction error over time produces
    Oudeyer-style "learning progress", which acts as an intrinsic reward for artificial curiosity.

    Attributes:
        R (np.ndarray): Fixed random projection matrix compressing 870D scene vectors to 64D gist (shape: (dim_g, X_DIM)).
        M (np.ndarray): Sensorimotor transition prediction matrix mapping (gist + action) to next gist (shape: (dim_g, dim_g + N_ACTIONS)).
        fast (float): Fast-decaying exponential moving average of prediction error.
        slow (float): Slow-decaying exponential moving average of prediction error.
    """

    def __init__(self, dim_g: int = 64, seed: int = 2) -> None:
        """
        Initializes the predictive forward model with a random projection matrix.

        Args:
            dim_g (int): Dimensionality of the compressed sensory gist representation (default: 64).
            seed (int): Seed for random projection initialization.
        """
        rng = np.random.RandomState(seed)
        # Random projection matrix normalized by 1 / sqrt(X_DIM) (Johnson-Lindenstrauss lemma)
        self.R: np.ndarray = (rng.randn(dim_g, X_DIM) / math.sqrt(X_DIM)).astype(np.float32)
        # Linear sensorimotor transition mapping weights
        self.M: np.ndarray = np.zeros((dim_g, dim_g + N_ACTIONS), dtype=np.float32)
        self.fast: float = 0.15
        self.slow: float = 0.15

    def gist(self, x: np.ndarray) -> np.ndarray:
        """
        Projects an 870D holistic scene vector down into a compact 64D unit-normalized latent gist.

        Args:
            x (np.ndarray): 870-dimensional scene vector.

        Returns:
            np.ndarray: 64-dimensional unit-normalized latent gist vector g.
        """
        g = self.R @ x
        return g / (np.linalg.norm(g) + 1e-8)

    def update(self, g: np.ndarray, a: int, g_next: np.ndarray, eta: float = 0.03) -> float:
        """
        Updates the forward transition matrix M via local Widrow-Hoff delta rule.

        Args:
            g (np.ndarray): Sensory gist vector at timestep t.
            a (int): Motor action executed at timestep t.
            g_next (np.ndarray): Resulting sensory gist vector observed at timestep t + 1.
            eta (float): Learning rate for transition weights M (default: 0.03).

        Returns:
            float: Mean absolute prediction error (epistemic surprise) before synaptic update.
        """
        # Construct sensorimotor input vector z: [g (64D), one_hot(a) (37D)] -> 101D
        z = np.zeros(self.M.shape[1], dtype=np.float32)
        z[:g.size] = g
        z[g.size + a] = 1.0

        # Prediction error: discrepancy between actual next gist and model anticipation
        predicted_next = self.M @ z
        error_vector = g_next - predicted_next

        # Local delta-rule update (Outer product: no backprop through layers)
        self.M += eta * np.outer(error_vector, z)

        # Compute scalar absolute error
        scalar_error = float(np.mean(np.abs(error_vector)))

        # Update fast and slow error tracking filters for learning progress computation
        self.fast += 0.3 * (scalar_error - self.fast)
        self.slow += 0.02 * (scalar_error - self.slow)

        return scalar_error

    @property
    def progress(self) -> float:
        """
        Evaluates learning progress: difference between slow and fast prediction error filters.

        A positive value (slow > fast) signifies that prediction error is falling—the agent is actively
        comprehending and mastering environmental dynamics, which triggers intrinsic reward.

        Returns:
            float: Learning progress metric.
        """
        return self.slow - self.fast


# =================================================================================================
# POLICY: TD(lambda) THREE-FACTOR HEBBIAN
# =================================================================================================
class TDHebbianPolicy:
    """
    TD(lambda) Three-Factor Hebbian reinforcement learning policy.

    Action-value function Q(s, a) is decomposed into biological components:
      Q(s, a) = sum_{i in active cortical units} W[a, i]   (sparse cortical voting)
              + V · f_cell(a)                               (shared local visual affordance for clicks)
              + U[a] · f_global                             (global visual affordance for non-spatial actions)

    Plasticity follows the canonical Three-Factor Rule:
      Delta W = (eta / k) * TD_error * Eligibility_Trace
      - Factor 1: Presynaptic activity (active cortical units / visual features).
      - Factor 2: Postsynaptic activity (action execution).
      - Factor 3: Neuromodulatory gating signal (temporal difference error delta = r + gamma * max Q' - Q).

    Completely gradient-free: single-layer linear associative learning with hard weight bounds.

    Attributes:
        W (np.ndarray): Cortical synaptic weights from sparse units to actions (shape: (N_ACTIONS, n_units)).
        V (np.ndarray): Shared visual affordance weights for spatial fixation clicks (shape: (N_LOC,)).
        U (np.ndarray): Visual affordance weights for global motor actions (shape: (N_ACTIONS - N_CELLS, N_LOC)).
        eW (np.ndarray): Eligibility trace accumulator for cortical weights W.
        eV (np.ndarray): Eligibility trace accumulator for local affordance weights V.
        eU (np.ndarray): Eligibility trace accumulator for global affordance weights U.
        gamma (float): Temporal discount factor for future rewards (default: 0.9).
        lam (float): Eligibility trace decay rate lambda (default: 0.6).
        eta (float): Learning rate for cortical weights W (default: 0.15).
        eta_v (float): Learning rate for affordance weights V and U (default: 0.05).
        k (int): Number of winning units in sparse code.
        last_delta (float): Most recent temporal difference error signal.
    """

    def __init__(self, n_units: int, k: int, gamma: float = 0.9, lam: float = 0.6,
                 eta: float = 0.15, eta_v: float = 0.05) -> None:
        """
        Initializes policy weights, eligibility traces, and hyperparameter coefficients.

        Args:
            n_units (int): Number of cortical units in sparse code (e.g., 2048).
            k (int): Sparsity level (number of active units, e.g., 24).
            gamma (float): TD reward discount factor (default: 0.9).
            lam (float): Eligibility trace decay parameter (default: 0.6).
            eta (float): Cortical weight learning rate (default: 0.15).
            eta_v (float): Affordance feature learning rate (default: 0.05).
        """
        self.W: np.ndarray = np.zeros((N_ACTIONS, n_units), dtype=np.float32)
        self.V: np.ndarray = np.zeros(N_LOC, dtype=np.float32)
        self.U: np.ndarray = np.zeros((N_ACTIONS - N_CELLS, N_LOC), dtype=np.float32)
        self.eW: np.ndarray = np.zeros_like(self.W)
        self.eV: np.ndarray = np.zeros_like(self.V)
        self.eU: np.ndarray = np.zeros_like(self.U)
        self.gamma: float = gamma
        self.lam: float = lam
        self.eta: float = eta
        self.eta_v: float = eta_v
        self.k: int = k
        self.last_delta: float = 0.0

    def q(self, idx: np.ndarray, F: np.ndarray, gf: np.ndarray) -> np.ndarray:
        """
        Evaluates the expected return Q(s, a) across all discrete actions for state s = (idx, F, gf).

        Args:
            idx (np.ndarray): Indices of the k active cortical units (shape: (k,)).
            F (np.ndarray): Local affordance feature matrix for all 32 click cells (shape: (N_CELLS, N_LOC)).
            gf (np.ndarray): Global visual feature vector (spatial average across all cells; shape: (N_LOC,)).

        Returns:
            np.ndarray: Vector of action values Q of shape (N_ACTIONS,), float32.
        """
        # Sum synaptic weights from all active cortical units for each action
        q_values = self.W[:, idx].sum(axis=1)

        # For spatial click actions (0..31): add dot product with shared local visual filter V
        q_values[:N_CELLS] += F @ self.V

        # For global actions (32..36: scroll, read, type, back): add dot product with action-specific filter U
        q_values[N_CELLS:] += self.U @ gf

        return q_values

    @staticmethod
    def select(q: np.ndarray, eps: float, tau: float, rng: np.random.RandomState) -> int:
        """
        Selects an action using Boltzmann-Gibbs softmax exploration with an epsilon-greedy exploratory floor.

        Args:
            q (np.ndarray): Action-value vector of shape (N_ACTIONS,).
            eps (float): Epsilon-greedy random exploration probability floor in [0.0, 1.0].
            tau (float): Softmax temperature (lower values exploit; higher values explore uniformly).
            rng (np.random.RandomState): Seeded random number generator.

        Returns:
            int: Selected discrete action index in [0, N_ACTIONS - 1].
        """
        # Epsilon-greedy exploratory branch
        if rng.rand() < eps:
            return int(rng.randint(N_ACTIONS))

        # Numerically stable Boltzmann-Gibbs softmax distribution
        z = (q - q.max()) / max(tau, 1e-3)
        probabilities = np.exp(np.clip(z, -30.0, 0.0))
        probabilities /= probabilities.sum()

        return int(rng.choice(N_ACTIONS, p=probabilities))

    def update(self, st: Tuple[np.ndarray, np.ndarray, np.ndarray], a: int, r: float,
               st_next: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Executes an online three-factor TD(lambda) Hebbian synaptic weight update.

        Args:
            st (Tuple): Current state representations (idx, F, gf).
            a (int): Motor action executed.
            r (float): Scalar reinforcement reward signal received.
            st_next (Tuple): Next state representations (idx_next, F_next, gf_next).
        """
        idx, F, gf = st
        q_current = float(self.q(idx, F, gf)[a])
        q_next_max = float(self.q(*st_next).max())

        # TD Error: delta = r + gamma * max Q(s', a') - Q(s, a)
        delta = r + self.gamma * q_next_max - q_current
        self.last_delta = delta

        # Decay eligibility traces geometrically: e <- gamma * lambda * e
        trace_decay = self.gamma * self.lam
        self.eW *= trace_decay
        self.eV *= trace_decay
        self.eU *= trace_decay

        # Accumulate eligibility traces for the executed action
        self.eW[a, idx] += 1.0
        if a < N_CELLS:
            self.eV += F[a]
        else:
            self.eU[a - N_CELLS] += gf

        # Three-Factor Weight Updates (Plasticity modulated by neuromodulatory TD error)
        self.W += (self.eta / self.k) * delta * self.eW
        self.V += self.eta_v * delta * self.eV
        self.U += self.eta_v * delta * self.eU

        # Bound weights to prevent explosive runaway excitation
        np.clip(self.W, -4.0, 4.0, out=self.W)
        np.clip(self.V, -4.0, 4.0, out=self.V)
        np.clip(self.U, -4.0, 4.0, out=self.U)

    def replay_update(self, st: Tuple[np.ndarray, np.ndarray, np.ndarray], a: int, r: float,
                      st_next: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Offline sleep replay update: one-step TD update without eligibility trace accumulation.

        Args:
            st (Tuple): Replayed state tuple (idx, F, gf).
            a (int): Replayed action.
            r (float): Replayed reward.
            st_next (Tuple): Replayed successor state tuple.
        """
        idx, F, gf = st
        q_current = float(self.q(idx, F, gf)[a])
        q_next_max = float(self.q(*st_next).max())
        delta = r + self.gamma * q_next_max - q_current

        self.W[a, idx] += (self.eta / self.k) * delta
        if a < N_CELLS:
            self.V += self.eta_v * delta * F[a]
        else:
            self.U[a - N_CELLS] += self.eta_v * delta * gf


# =================================================================================================
# LANGUAGE: HEBBIAN N-GRAM CORTEX
# =================================================================================================
class LanguageCortex:
    """
    Hebbian associative N-gram and semantic cortex.

    Constructs language associations incrementally from raw sensory text:
      1. Vocabulary Acquisition: Dynamically builds word-to-id mappings capped at 30,000 words.
      2. Surprisal-Gated Plasticity: Computes information-theoretic surprisal via smoothed bigrams.
         Predictable text (low surprisal) is habituated; surprising text receives boosted synaptic learning.
      3. Tri-Gram and Bi-Gram Transitions: Models syntactic sequence dependencies.
      4. Semantic Co-Occurrence Graph: Captures associative semantic relatedness within a sliding window (+/- 4 words).
      5. Spreading Activation: Computes thematic topical context across semantic networks.
      6. Inner Speech Generation: Blends trigrams, bigrams, and spreading activation into internal monologue.
      7. Homeostatic Synaptic Decay: Prunes weak or unused associations during sleep consolidation.

    Class Attributes:
        FUNC (frozenset): Set of closed-class grammatical function words excluded from semantic concepts.
        BOILER (Tuple[str, ...]): Common boilerplate phrases filtered out during web text digestion.

    Attributes:
        w2i (Dict[str, int]): Mapping from word strings to unique integer vocabulary IDs.
        i2w (Dict[int, str]): Inverted mapping from integer IDs back to word strings.
        count (Dict[int, int]): Total observation count for each word ID.
        bi (Dict[int, Dict[int, float]]): Bigram transition weights {w1_id: {w2_id: weight}}.
        bi_tot (Dict[int, float]): Sum of outgoing bigram weights from each source word for probability normalization.
        tri (Dict[Tuple[int, int], Dict[int, float]]): Trigram transition weights {(w1_id, w2_id): {w3_id: weight}}.
        co (Dict[int, Dict[int, float]]): Semantic co-occurrence matrix within sliding window.
        eta (float): Synaptic potentiation step size for language associations (default: 0.25).
    """

    FUNC: frozenset = frozenset(
        "the a an of in to and is was that for on with as by at from it can cannot could would should "
        "will have has had he she they we i you her his my our their so but or one are be this these "
        "those not no been were which also its than then".split()
    )
    BOILER: Tuple[str, ...] = (
        "jump to", "navigation", "cookie", "privacy policy", "terms of use", "disclaimer", "contact us",
        "donate", "view history", "edit this", "download as", "sign in", "log in", "subscribe",
        "all rights reserved", "main menu", "toggle", "skip to"
    )

    def __init__(self) -> None:
        """Initializes the language cortex data structures and special tokens."""
        self.w2i: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.i2w: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.count: Dict[int, int] = {}
        self.bi: Dict[int, Dict[int, float]] = {}
        self.bi_tot: Dict[int, float] = {}
        self.tri: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.co: Dict[int, Dict[int, float]] = {}
        self.eta: float = 0.25

    @staticmethod
    def words_of(text: str) -> List[str]:
        """
        Tokenizes an input string into lowercase alphanumeric words.

        Args:
            text (str): Input text string.

        Returns:
            List[str]: List of extracted lowercase word tokens.
        """
        return re.findall(r"[a-z][a-z0-9']*", text.lower())

    def sentences(self, text: str) -> List[str]:
        """
        Cleans and splits raw text into valid grammatical sentences, discarding navigation boilerplate.

        Args:
            text (str): Raw string parsed from visual/sensory environment.

        Returns:
            List[str]: Cleaned list of natural sentence strings suitable for cognitive processing.
        """
        normalized = re.sub(r"\s*\n\s*", " ", text)
        sentences_out: List[str] = []

        # Split on standard sentence terminal punctuation followed by whitespace
        for s in re.split(r"(?<=[.!?])\s+", normalized):
            tokens = s.split()
            # Enforce reasonable sentence lengths (4 to 45 words)
            if not 4 <= len(tokens) <= 45:
                continue
            # Ensure high ratio of standard alphabetical tokens
            alpha_count = sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z'\-,.;:()]*", t)) for t in tokens)
            if alpha_count / len(tokens) < 0.7 or any(boilerplate in s.lower() for boilerplate in self.BOILER):
                continue
            sentences_out.append(s)

        return sentences_out

    def _id(self, w: str, create: bool = True) -> int:
        """
        Looks up or registers a word in the vocabulary dictionary.

        Args:
            w (str): Word token string.
            create (bool): If True, assigns a new ID if word is unmapped and vocab capacity allows.

        Returns:
            int: Integer vocabulary ID (or 1 for <UNK> if capacity is reached).
        """
        if w in self.w2i:
            return self.w2i[w]
        if not create or len(self.w2i) >= 30000:
            return 1  # <UNK>
        new_id = len(self.w2i)
        self.w2i[w] = new_id
        self.i2w[new_id] = w
        return new_id

    def surprisal(self, words: List[str]) -> float:
        """
        Computes information-theoretic surprisal (bits per transition) for a sequence of words.

        Evaluated under a Laplace-smoothed bigram probability model:
          P(w_{t} | w_{t-1}) = (count(w_{t-1}, w_{t}) + eps / V) / (total(w_{t-1}) + eps)
          Surprisal = -log2 P(w_{t} | w_{t-1})

        Args:
            words (List[str]): Sequence of word tokens.

        Returns:
            float: Average surprisal in bits per transition. High values denote unfamiliar/unexpected sequences.
        """
        if len(words) < 2:
            return 0.0

        # Floor vocabulary size V at 2000 so that an empty cortex finds early text maximally surprising
        vocab_size = max(len(self.w2i) + 1, 2000)
        eps = 0.5
        total_surprisal = 0.0

        for w1, w2 in zip(words, words[1:]):
            i1 = self.w2i.get(w1, -1)
            i2 = self.w2i.get(w2, -1)
            row = self.bi.get(i1, {})
            # Smoothed conditional probability
            prob = (row.get(i2, 0.0) + eps / vocab_size) / (self.bi_tot.get(i1, 0.0) + eps)
            total_surprisal += -math.log2(prob)

        return total_surprisal / (len(words) - 1)

    def ingest(self, sentence: str, strength: float = 1.0) -> int:
        """
        Incorporates a sentence into lexical, bigram, trigram, and semantic co-occurrence networks.

        Args:
            sentence (str): Sentence string to absorb.
            strength (float): Plasticity modulation factor (amplified by high surprisal or human interaction).

        Returns:
            int: Number of brand-new words incorporated into vocabulary.
        """
        words = self.words_of(sentence)
        if len(words) < 2:
            return 0

        new_word_count = sum(1 for w in words if w not in self.w2i)
        ids = [self._id(w) for w in words]

        # 1. Update unigram frequencies
        for word_id in ids:
            self.count[word_id] = self.count.get(word_id, 0) + 1

        # 2. Update bigram forward associations
        for a, b in zip(ids, ids[1:]):
            self.bi.setdefault(a, {})
            self.bi[a][b] = self.bi[a].get(b, 0.0) + self.eta * strength
            self.bi_tot[a] = self.bi_tot.get(a, 0.0) + self.eta * strength

        # 3. Update trigram sequential associations
        for a, b, c in zip(ids, ids[1:], ids[2:]):
            trigram_row = self.tri.setdefault((a, b), {})
            trigram_row[c] = trigram_row.get(c, 0.0) + self.eta * strength * 2.5

        # 4. Update semantic co-occurrence network (+/- 4 words window)
        for pos, a in enumerate(ids):
            co_row = self.co.setdefault(a, {})
            window_start = max(0, pos - 4)
            window_end = min(len(ids), pos + 5)
            for neighbor_pos in range(window_start, window_end):
                if neighbor_pos != pos:
                    neighbor_id = ids[neighbor_pos]
                    co_row[neighbor_id] = co_row.get(neighbor_id, 0.0) + self.eta * 0.4 * strength

        return new_word_count

    def decay(self, f: float = 0.97, floor: float = 0.05) -> None:
        """
        Homeostatic forgetting during sleep: attenuates synaptic weights and prunes sub-threshold connections.

        Args:
            f (float): Synaptic decay multiplier (default: 0.97).
            floor (float): Absolute weight threshold below which an association is pruned (default: 0.05).
        """
        # Decay bigram and co-occurrence graphs
        for table in (self.bi, self.co):
            for source in list(table):
                dest_dict = table[source]
                for dest in list(dest_dict):
                    dest_dict[dest] *= f
                    if dest_dict[dest] < floor:
                        del dest_dict[dest]
                if not dest_dict:
                    del table[source]

        # Decay trigram graph
        for context_tuple in list(self.tri):
            dest_dict = self.tri[context_tuple]
            for dest in list(dest_dict):
                dest_dict[dest] *= f
                if dest_dict[dest] < floor:
                    del dest_dict[dest]
            if not dest_dict:
                del self.tri[context_tuple]

        # Recompute bigram normalization sums
        self.bi_tot = {source: sum(dests.values()) for source, dests in self.bi.items()}

    def concepts(self, words: List[str]) -> List[str]:
        """
        Filters word tokens to extract meaningful conceptual nouns/verbs, excluding function/stop words.

        Args:
            words (List[str]): List of word tokens.

        Returns:
            List[str]: Filtered list of semantically rich content words known to vocabulary.
        """
        return [w for w in words if w not in self.FUNC and w in self.w2i]

    def _spread(self, topic: set) -> Dict[int, float]:
        """
        Propagates spreading activation across the semantic co-occurrence network given topical seeds.

        Args:
            topic (set): Set of integer word IDs representing current conceptual focus.

        Returns:
            Dict[int, float]: Normalized dictionary mapping associated word IDs to their spreading activation levels.
        """
        activation: Dict[int, float] = {}
        for term_id in topic:
            for neighbor_id, weight in self.co.get(term_id, {}).items():
                if self.i2w.get(neighbor_id) not in self.FUNC:
                    activation[neighbor_id] = activation.get(neighbor_id, 0.0) + weight

        total_activation = sum(activation.values()) + 1e-6
        return {word_id: act / total_activation for word_id, act in activation.items()}

    def generate(self, seeds: List[str], n: int = 12, temp: float = 0.6) -> str:
        """
        Generates spontaneous inner monologue utterances by blending trigrams, bigrams, and spreading activation.

        Args:
            seeds (List[str]): Seed words to initiate the thought sequence.
            n (int): Maximum number of words to generate (default: 12).
            temp (float): Softmax sampling temperature for diversity control (default: 0.6).

        Returns:
            str: Generated string of sequential thought tokens.
        """
        seed_ids = [self.w2i[w] for w in seeds if w in self.w2i]
        if not seed_ids:
            return " ".join(seeds)

        generated_ids = list(seed_ids)
        topical_focus = {word_id for word_id in seed_ids if self.i2w[word_id] not in self.FUNC}
        spread_activation = self._spread(topical_focus)

        for _ in range(n):
            current_id = generated_ids[-1]
            previous_id = generated_ids[-2] if len(generated_ids) > 1 else None
            candidates: Dict[int, float] = {}

            # 1. Trigram candidate transitions
            if previous_id is not None:
                trigram_row = self.tri.get((previous_id, current_id))
                if trigram_row:
                    total_trigram = sum(trigram_row.values()) + 1e-6
                    for next_word, weight in trigram_row.items():
                        candidates[next_word] = candidates.get(next_word, 0.0) + 4.0 * weight / total_trigram

            # 2. Bigram candidate transitions
            bigram_row = self.bi.get(current_id)
            if bigram_row:
                total_bigram = self.bi_tot.get(current_id, 0.0) + 1e-6
                scale = 1.0 if candidates else 2.5
                for next_word, weight in bigram_row.items():
                    candidates[next_word] = candidates.get(next_word, 0.0) + scale * weight / total_bigram

            if not candidates:
                break

            # 3. Modulate candidate probabilities by semantic spreading activation and repetition penalties
            for candidate_id in candidates:
                repetition_count = generated_ids.count(candidate_id)
                topical_boost = 1.0 + 30.0 * spread_activation.get(candidate_id, 0.0)
                repetition_penalty = 1.0 + 3.0 * repetition_count
                candidates[candidate_id] *= topical_boost / repetition_penalty

            # Sample next word from top-12 candidates according to temperature
            top_candidates = sorted(candidates.items(), key=lambda kv: -kv[1])[:12]
            weights = np.array([val for _, val in top_candidates], dtype=np.float64) ** (1.0 / temp)
            probabilities = weights / weights.sum()

            next_word_index = int(np.random.choice(len(top_candidates), p=probabilities))
            selected_word_id = top_candidates[next_word_index][0]

            generated_ids.append(selected_word_id)

            if self.i2w[selected_word_id] not in self.FUNC:
                topical_focus.add(selected_word_id)
                spread_activation = self._spread(topical_focus)

        return " ".join(self.i2w[word_id] for word_id in generated_ids)

    def save(self, path: str) -> None:
        """
        Serializes the language cortex graph structures to compressed NPZ format.

        Args:
            path (str): File destination path (.npz).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        words = list(self.w2i)

        flatten_table = lambda tbl: [(s, d, v) for s, row in tbl.items() for d, v in row.items()]
        bi_flat = flatten_table(self.bi)
        co_flat = flatten_table(self.co)
        tri_flat = [(a, b, d, v) for (a, b), row in self.tri.items() for d, v in row.items()]

        make_array = lambda rows, col, dtype: np.array([r[col] for r in rows], dtype=dtype) if rows else np.empty(0, dtype=dtype)

        tmp_path = path + ".tmp.npz"
        np.savez_compressed(
            tmp_path,
            words=np.array(words),
            ids=np.array([self.w2i[w] for w in words], dtype=np.int32),
            counts=np.array([self.count.get(self.w2i[w], 0) for w in words], dtype=np.int32),
            bi_src=make_array(bi_flat, 0, np.int32),
            bi_dst=make_array(bi_flat, 1, np.int32),
            bi_val=make_array(bi_flat, 2, np.float32),
            tri_s1=make_array(tri_flat, 0, np.int32),
            tri_s2=make_array(tri_flat, 1, np.int32),
            tri_dst=make_array(tri_flat, 2, np.int32),
            tri_val=make_array(tri_flat, 3, np.float32),
            co_src=make_array(co_flat, 0, np.int32),
            co_dst=make_array(co_flat, 1, np.int32),
            co_val=make_array(co_flat, 2, np.float32)
        )
        os.replace(tmp_path, path)

    def load(self, path: str) -> bool:
        """
        Restores the language cortex graph from a saved NPZ file.

        Args:
            path (str): File path to saved .npz checkpoint.

        Returns:
            bool: True if successfully loaded, False otherwise.
        """
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)
            self.w2i = {str(w): int(i) for w, i in zip(data["words"], data["ids"])}
            self.i2w = {i: w for w, i in self.w2i.items()}
            self.count = {int(i): int(c) for i, c in zip(data["ids"], data["counts"])}
            self.bi, self.tri, self.co = {}, {}, {}

            for s, t, v in zip(data["bi_src"], data["bi_dst"], data["bi_val"]):
                self.bi.setdefault(int(s), {})[int(t)] = float(v)

            for a, b, c, v in zip(data["tri_s1"], data["tri_s2"], data["tri_dst"], data["tri_val"]):
                self.tri.setdefault((int(a), int(b)), {})[int(c)] = float(v)

            if "co_src" in data:
                for s, t, v in zip(data["co_src"], data["co_dst"], data["co_val"]):
                    self.co.setdefault(int(s), {})[int(t)] = float(v)

            self.bi_tot = {s: sum(r.values()) for s, r in self.bi.items()}
            return True
        except Exception:
            return False


# =================================================================================================
# AWARENESS MACHINERY: DRIVES, GLOBAL WORKSPACE, SELF-MODEL
# =================================================================================================
class Drives:
    """
    Homeostatic motivational drive engine.

    Models biological drive states that regulate autonomous behavior:
      1. Boredom: Accumulates when sensory novelty is persistently low.
      2. Curiosity: Directly derived from boredom; increases willingness to explore novel stimuli.
      3. Fatigue: Accumulates linearly per cognitive action step; triggers sleep consolidation at 1.0.
      4. Social Engagement: Heightened by human communication; decays exponentially over time.

    Attributes:
        boredom (float): Current boredom drive level in [0.0, 1.0].
        fatigue (float): Cumulative cognitive fatigue in [0.0, 1.0].
        social (float): Current social engagement motivation in [0.0, 1.0].
        sleep_every (int): Number of cognitive steps before fatigue reaches 1.0.
    """

    def __init__(self, sleep_every: int = 400) -> None:
        """
        Initializes homeostatic drive levels.

        Args:
            sleep_every (int): Timestep duration of wakefulness cycle (default: 400).
        """
        self.boredom: float = 0.5
        self.fatigue: float = 0.0
        self.social: float = 0.0
        self.sleep_every: int = sleep_every

    @property
    def curiosity(self) -> float:
        """
        Evaluates current curiosity level derived from boredom.

        Returns:
            float: Curiosity metric bounded in [0.15, 1.0].
        """
        return float(np.clip(0.15 + 0.85 * self.boredom, 0.15, 1.0))

    def update(self, novelty: float, human: bool) -> None:
        """
        Updates internal drives based on sensory novelty and social interactions.

        Args:
            novelty (float): Perceptual novelty score from episodic memory in [0.0, 1.0].
            human (bool): True if human communicative input was received this timestep.
        """
        # Boredom rises when novelty is low, falls when experiencing novel environments
        target_boredom = 1.0 - min(1.0, novelty * 2.0)
        self.boredom += 0.03 * (target_boredom - self.boredom)

        # Fatigue increases steadily toward circadian sleep requirement
        self.fatigue += 1.0 / self.sleep_every

        # Social drive surges upon human interaction, decays otherwise
        self.social = 1.0 if human else self.social * 0.92

    def exploration(self, confidence: float) -> Tuple[float, float]:
        """
        Calculates dynamic exploration hyperparameters for action selection.

        Args:
            confidence (float): Subjective metacognitive confidence metric in [0.0, 1.0].

        Returns:
            Tuple[float, float]:
                - eps (float): Epsilon-greedy random exploration probability floor.
                - tau (float): Boltzmann-Gibbs softmax exploration temperature.
        """
        c = self.curiosity
        # Epsilon scales with curiosity and inversely with confidence
        eps = 0.04 + 0.25 * c * (1.0 - 0.5 * confidence)
        # Softmax temperature scales with curiosity
        tau = 0.12 + 0.25 * c
        return eps, tau

    def rest(self) -> None:
        """Resets fatigue to zero upon completion of sleep consolidation."""
        self.fatigue = 0.0


class GlobalWorkspace:
    """
    Global Workspace Theory (GWT) working memory stage.

    Implements a capacity-limited conscious bottleneck where candidate sensory, cognitive,
    and motivational representations compete based on salience. Winning items are broadcast
    system-wide to direct internal monologue and action selection.

    Attributes:
        items (List[Dict[str, Any]]): List of mental items currently competing on stage.
        capacity (int): Maximum number of concurrent items allowed on stage (default: 4).
    """

    def __init__(self, capacity: int = 4) -> None:
        """
        Initializes the working memory stage.

        Args:
            capacity (int): Maximum capacity of the workspace bottleneck (default: 4).
        """
        self.items: List[Dict[str, Any]] = []
        self.capacity: int = capacity

    def post(self, src: str, words: List[str], sal: float) -> None:
        """
        Submits a candidate mental representation to compete in the workspace.

        Args:
            src (str): Originating subsystem tag (e.g., 'novelty', 'text', 'human').
            words (List[str]): Conceptual semantic tokens associated with the item.
            sal (float): Initial salience intensity score in [0.0, 1.0].
        """
        self.items.append({"src": src, "words": list(words), "sal": float(sal)})

    def tick(self) -> None:
        """
        Decays the salience of active items, purges sub-threshold items, and truncates to capacity.
        """
        # Exponential salience decay
        for item in self.items:
            item["sal"] *= 0.85

        # Retain items above threshold (0.05) sorted descending by salience up to capacity
        self.items = sorted((item for item in self.items if item["sal"] > 0.05),
                            key=lambda item: -item["sal"])[:self.capacity]

    def winner(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the dominant conscious item winning the workspace competition.

        Returns:
            Optional[Dict[str, Any]]: Most salient workspace item, or None if empty.
        """
        return self.items[0] if self.items else None


class SelfModel:
    """
    Metacognitive self-monitoring module.

    Maintains running internal statistics regarding the organism's own performance,
    prediction accuracy, and behavioral distribution:
      1. Running Reward (ema_r): Moving average of received reinforcement.
      2. Encountered Novelty (ema_nov): Moving average of perceptual novelty.
      3. Prediction Error (ema_err): Moving average of forward model prediction error.
      4. Subjective Confidence: Inversely related to prediction error.
      5. Behavioral Action Entropy: Shannon entropy of the organism's motor execution distribution.

    Attributes:
        ema_r (float): Exponential moving average of reinforcement rewards.
        ema_nov (float): Exponential moving average of perceptual novelty.
        ema_err (float): Exponential moving average of forward model prediction errors.
        counts (np.ndarray): Frequency tally across all discrete action types (shape: (N_ACTIONS,)).
    """

    def __init__(self) -> None:
        """Initializes metacognitive running statistics."""
        self.ema_r: float = 0.0
        self.ema_nov: float = 0.5
        self.ema_err: float = 0.15
        self.counts: np.ndarray = np.ones(N_ACTIONS, dtype=np.float32)

    def observe(self, a: int, r: float, nov: float, err: float) -> None:
        """
        Updates internal running statistics following a cognitive step.

        Args:
            a (int): Action executed.
            r (float): Reward received.
            nov (float): Sensory novelty observed.
            err (float): Forward model prediction error.
        """
        self.ema_r += 0.02 * (r - self.ema_r)
        self.ema_nov += 0.02 * (nov - self.ema_nov)
        self.ema_err += 0.02 * (err - self.ema_err)
        self.counts[a] += 1

    @property
    def confidence(self) -> float:
        """
        Calculates subjective confidence inversely proportional to forward prediction error.

        Returns:
            float: Confidence metric bounded in (0.0, 1.0].
        """
        return 1.0 / (1.0 + 10.0 * self.ema_err)

    def report(self) -> Dict[str, float]:
        """
        Compiles a summary telemetry report of internal metacognitive metrics.

        Returns:
            Dict[str, float]: Telemetry dictionary containing avg_reward, novelty_rate, confidence, and behaviour_entropy.
        """
        prob_dist = self.counts / self.counts.sum()
        # Normalized Shannon entropy across discrete actions: H(X) / log(N_ACTIONS)
        entropy = float(-(prob_dist * np.log(prob_dist)).sum() / math.log(N_ACTIONS))
        return {
            "avg_reward": round(self.ema_r, 3),
            "novelty_rate": round(self.ema_nov, 3),
            "confidence": round(self.confidence, 3),
            "behaviour_entropy": round(entropy, 3)
        }


# =================================================================================================
# THE ORGANISM (CORE COGNITIVE AGENT)
# =================================================================================================
class Organism:
    """
    Unified cognitive agent integrating visual perception, sparse coding, episodic memory,
    predictive forward modeling, TD-Hebbian reinforcement learning, homeostatic drives,
    global workspace broadcasting, associative language cortex, and metacognition.

    Attributes:
        env (Any): Environmental interface providing sensory observations and accepting motor actions.
        verbose (bool): If True, prints formatted cognitive telemetry to stdout each step.
        explore_only (bool): If True, enforces purely random exploration (used as experimental baseline).
        rng (np.random.RandomState): Seeded pseudo-random number generator for reproducibility.
        ckpt (str): File path for saving/loading network weights and episodic prototypes (.npz).
        lang_path (str): File path for saving/loading language cortex associations (.npz).
        autosave_every (int): Timestep interval between automatic disk checkpoint writes.
        retina (Retina): Biological receptive field sensory encoder.
        encoder (SparseEncoder): Competitive k-WTA cortical representation layer.
        mem (EpisodicMemory): Exemplar prototype memory bank with one-shot credit assignment.
        fwd (ForwardModel): Predictive forward sensorimotor transition model.
        policy (TDHebbianPolicy): Three-factor TD(lambda) Hebbian reinforcement learner.
        lang (LanguageCortex): Associative lexical, syntactic, and semantic language engine.
        drives (Drives): Homeostatic motivational drive controller (boredom, fatigue, social).
        ws (GlobalWorkspace): Capacity-limited working memory competition stage.
        me (SelfModel): Metacognitive self-monitoring module.
        replay (deque): Bounded episodic transition buffer utilized during sleep replay.
        dead (Dict[int, set]): Per-prototype dead-action set enforcing scene-specific inhibition of return.
        prev (Optional[Dict[str, Any]]): Cached state representations and chosen action from previous timestep.
        t (int): Global lifetime cognitive step counter.
        last_read_t (int): Timestep when textual reading was last requested.
        sleeps (int): Total count of sleep consolidation cycles completed.
        thought (str): Current internal monologue utterance generated by language cortex.
        last_concepts (List[str]): Most recent salient concept words extracted from sensory text.
        pending_fb (float): Pending scalar human feedback credit awaiting integration.
        human_msgs (queue.Queue): Thread-safe queue receiving incoming human communicative text.
        replies (queue.Queue[str]): Thread-safe queue containing generated verbal responses to human input.
        running (bool): Flag indicating active cognitive loop execution status.
        stats (Dict[str, Any]): Cumulative runtime statistics (clicks, effective transitions, rewards).
    """

    def __init__(self, env: Any = None, ckpt_dir: str = CKPT_DIR, seed: int = 0, sleep_every: int = 400,
                 explore_only: bool = False, verbose: bool = True, autosave_every: int = 200,
                 legacy_language: Optional[str] = None) -> None:
        """
        Initializes the unified cognitive organism and loads saved checkpoints if present.

        Args:
            env (Any): Simulation or robotic environment interface (optional).
            ckpt_dir (str): Directory path for checkpoint persistence.
            seed (int): Global deterministic random seed.
            sleep_every (int): Timestep interval between sleep consolidation phases (default: 400).
            explore_only (bool): If True, disables learned policy in favor of pure random exploration.
            verbose (bool): If True, prints telemetry log each cycle.
            autosave_every (int): Frequency in steps for autosaving state to disk (0 to disable).
            legacy_language (Optional[str]): Path to legacy language NPZ file for migration.
        """
        self.env: Any = env
        self.verbose: bool = verbose
        self.explore_only: bool = explore_only
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.ckpt: str = os.path.join(ckpt_dir, "organism_v2.npz")
        self.lang_path: str = os.path.join(ckpt_dir, "organism_v2_language.npz")
        self.autosave_every: int = autosave_every

        # Core Cognitive Subsystems
        self.retina: Retina = Retina()
        self.encoder: SparseEncoder = SparseEncoder()
        self.mem: EpisodicMemory = EpisodicMemory()
        self.fwd: ForwardModel = ForwardModel()
        self.policy: TDHebbianPolicy = TDHebbianPolicy(self.encoder.n_units, self.encoder.k)
        self.lang: LanguageCortex = LanguageCortex()
        self.drives: Drives = Drives(sleep_every)
        self.ws: GlobalWorkspace = GlobalWorkspace()
        self.me: SelfModel = SelfModel()

        # Episodic Experience Buffer and Working State
        self.replay: deque = deque(maxlen=3000)
        self.dead: Dict[int, set] = {}
        self.prev: Optional[Dict[str, Any]] = None
        self.t: int = 0
        self.last_read_t: int = -99
        self.sleeps: int = 0
        self.thought: str = ""
        self.last_concepts: List[str] = []

        # Human Interaction Interfaces (Thread-Safe)
        self.pending_fb: float = 0.0
        self.human_msgs: queue.Queue = queue.Queue()
        self.replies: queue.Queue[str] = queue.Queue()
        self.running: bool = False

        # Cumulative Telemetry
        self.stats: Dict[str, Any] = {
            "clicks": 0,
            "effective": 0,
            "reward_sum": 0.0,
            "novel_arrivals": 0
        }

        # Load weights and associative memories from checkpoint if available
        self.load(legacy_language)

    # ---------------------------------------------------------------------------------------------
    # HUMAN INTERACTION INTERFACE (Thread-Safe)
    # ---------------------------------------------------------------------------------------------
    def hear(self, text: str) -> None:
        """
        Thread-safe entrypoint for receiving spoken language or chat messages from a human user.

        Args:
            text (str): Incoming human communicative utterance.
        """
        self.human_msgs.put(text)

    def feedback(self, val: float) -> None:
        """
        Thread-safe entrypoint for applying scalar reinforcement feedback (praise or reprimand).

        Args:
            val (float): Scalar feedback value (e.g., +1.0 for approval, -1.0 for correction).
        """
        self.pending_fb += val

    # ---------------------------------------------------------------------------------------------
    # COGNITIVE CYCLE STEP
    # ---------------------------------------------------------------------------------------------
    def step(self, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes one full cognitive perception-cognition-action cycle.

        Cycle Stages:
          1. Observation Ingestion: Captures visual frame and motion flux from environment or parameter.
          2. Retinal Feature Extraction: Computes 6-channel maps, 870D pooled scene vector, and 8D local cell features.
          3. Sparse Cortical Encoding: Transforms scene vector into 24-winner active unit indices.
          4. Episodic Memory Recall: Queries prototype memory for matching prototype, novelty, and similarity.
          5. Predictive Forward Model: Compares previous prediction with new gist to evaluate epistemic surprise.
          6. Multi-Factor Reward Formulation: Blends novelty, perceptual change, reading surprisal, learning progress,
             and external feedback, penalized by wasted effort.
          7. Policy & Value Update: Three-factor Hebbian update modulated by TD(lambda) error.
          8. Motivational Regulation: Updates boredom, fatigue, and curiosity drives.
          9. Working Memory Broadcast: Competes items in Global Workspace; winner seeds inner speech.
          10. Action Selection: Softmax exploration balances affordance value Q against exploratory drives.
          11. Motor Execution: Triggers selected physical or digital action in environment.

        Args:
            obs (Optional[Dict[str, Any]]): Sensory observation dictionary with keys 'rgb', 'motion', and optional 'text'.
                                            If None, calls self.env.observe().

        Returns:
            Dict[str, Any]: Telemetry summary of the executed cognitive cycle.
        """
        self.t += 1
        if obs is None:
            if self.env is None:
                raise ValueError("No observation provided and no environment configured.")
            obs = self.env.observe()

        # 1. Visual Feature Extraction via Retina
        m6 = self.retina.encode(obs["rgb"], obs["motion"])
        x = self.retina.pooled(m6, obs["rgb"])
        F = self.retina.cell_features(m6, obs["rgb"])
        gf = F.mean(axis=0)

        # 2. Sparse Cortical Encoding & Forward Gist
        idx = self.encoder.encode(x, learn=True)
        g = self.fwd.gist(x)

        # 3. Episodic Prototype Query
        ctx, nov, sim = self.mem.observe(x, self.t)

        # 4. Multi-Modal Sensory & Social Digestion
        human = self._digest_human()
        raw_text = obs.get("text") if "text" in obs else (
            self.env.poll_text() if self.env and hasattr(self.env, "poll_text") else None
        )
        r_text = self._digest_text(raw_text)

        # Consume pending asynchronous human feedback
        fb, self.pending_fb = self.pending_fb, 0.0

        r = 0.0
        err = self.me.ema_err
        change = 0.0
        wasted = False

        # 5. Reinforcement & Learning from Previous Step Transition
        if self.prev is not None:
            p = self.prev
            # Compute epistemic forward model prediction error
            err = self.fwd.update(p["g"], p["a"], g)

            # Determine whether previous action produced substantial visual change
            if p["a"] != A_READ:
                cosine_sim_prev = float(x @ p["x"])
                change = float(np.clip((1.0 - cosine_sim_prev - 0.05) / 0.5, 0.0, 1.0))
                wasted = cosine_sim_prev > 0.985  # No visual change: wasted motor effort

            # Habituation discount on agency reward: diminishes on repeatedly visited scenes
            habit = 1.0 / (1.0 + 0.4 * float(self.mem.count[ctx]))

            # Intrinsic + Extrinsic Reward Formulation:
            #   - 0.55 * novelty: Curiosity incentive to visit unexplored sensory configurations.
            #   - 0.15 * change * habit: Sense-of-agency reward for exerting causal visual impact.
            #   - 0.40 * r_text: Semantic comprehension reward for acquiring novel lexical information.
            #   - 0.25 * progress: Intrinsic motivation for making predictive learning progress.
            #   - 0.08 wasted effort penalty: Deterrent against repeated dead-end clicks.
            #   - fb: Direct scalar human reward/punishment.
            r = (
                0.55 * nov +
                0.15 * change * habit +
                0.40 * r_text +
                0.25 * float(np.clip(self.fwd.progress * 5.0, -0.5, 0.5)) -
                (0.08 if wasted else 0.0) +
                fb
            )
            r = float(np.clip(r, -1.5, 1.5))

            # Current state representations tuple
            current_st = (idx, F, gf)

            # Policy and Episodic Value Updates
            self.policy.update(p["state"], p["a"], r, current_st)
            self.mem.credit(p["ctx"], p["a"], r)
            self.replay.append((p["state"], p["a"], r, current_st))
            self.me.observe(p["a"], r, nov, err)

            # Update performance statistics
            self.stats["reward_sum"] += r
            if p["a"] < N_CELLS:
                self.stats["clicks"] += 1
                self.stats["effective"] += int(change > 0.3)
                self.stats["novel_arrivals"] += int(change > 0.3 and nov > 0.5)

            # Inhibition of Return: if click produced zero visual change, suppress action on this prototype
            if wasted and p["a"] < N_CELLS:
                self.dead.setdefault(p["ctx"], set()).add(p["a"])

        # 6. Motivational Drives & Working Memory Update
        self.drives.update(nov, human)
        self.ws.tick()
        if nov > 0.3:
            self.ws.post("novelty", self.last_concepts[:4], nov)

        # Attentional capture: novel environments prompt cognitive reading request
        if nov > 0.35 and self.t - self.last_read_t > 12:
            if self.env and hasattr(self.env, "request_read"):
                self.env.request_read()
            self.last_read_t = self.t

        # 7. Action-Value Evaluation & Action Selection
        q = self.policy.q(idx, F, gf) + 0.5 * self.mem.Qe[ctx]

        # Apply inhibition of return penalty to actions known to be ineffective in this context
        for dead_action in self.dead.get(ctx, ()):
            q[dead_action] -= 0.3

        # Mask typing action if environment does not support textual input
        supports_typing = getattr(self.env, "supports_typing", False) if self.env else True
        if not supports_typing:
            q[A_TYPE] = -1e9

        # Dynamic exploration based on curiosity and metacognitive confidence
        eps, tau = self.drives.exploration(self.me.confidence)
        selected_action = self.policy.select(q, 1.0 if self.explore_only else eps, tau, self.rng)

        # Context-dependent text selection for typing actions
        typed_text = self._pick_text() if selected_action == A_TYPE else None

        # 8. Physical / Environmental Motor Action Execution
        action_info = (
            self.env.act(selected_action, text=typed_text)
            if self.env and hasattr(self.env, "act")
            else action_name(selected_action)
        )

        # 9. Internal Monologue / Inner Speech Generation
        if self.t % 6 == 0 or human or r_text > 0:
            workspace_winner = self.ws.winner()
            seed_concepts = (workspace_winner["words"] if workspace_winner and workspace_winner["words"] else self.last_concepts)[:3]
            self.thought = self.lang.generate(seed_concepts, n=10) if seed_concepts else ""

        # Cache current state for the subsequent transition learning step
        self.prev = {
            "state": (idx, F, gf),
            "a": selected_action,
            "g": g,
            "x": x,
            "ctx": ctx
        }

        # Periodic autosave
        if self.autosave_every and self.t % self.autosave_every == 0:
            self.save()

        return {
            "t": self.t,
            "a": selected_action,
            "info": action_info,
            "r": r,
            "nov": nov,
            "change": change,
            "pages": self.mem.n,
            "vocab": len(self.lang.w2i),
            "curiosity": self.drives.curiosity,
            "fatigue": self.drives.fatigue,
            "conf": self.me.confidence,
            "delta": self.policy.last_delta,
            "thought": self.thought
        }

    def _digest_text(self, txt: Optional[str]) -> float:
        """
        Parses sensory text from environment, computes surprisal, and ingests novel sentences.

        Args:
            txt (Optional[str]): Text string presented on current page.

        Returns:
            float: Normalized lexical discovery reward in [0.0, 1.0].
        """
        if not txt:
            return 0.0

        new_words = 0
        most_surprising = None

        for sentence in self.lang.sentences(txt):
            words = self.lang.words_of(sentence)
            surprisal_val = self.lang.surprisal(words)

            # Habituation: ignore sentences that are already fully predictable
            if surprisal_val < 2.0:
                continue

            # Ingest with plasticity strength scaled by surprisal
            ingested_count = self.lang.ingest(sentence, strength=1.0 + min(4.0, surprisal_val / 2.0))
            new_words += ingested_count

            if most_surprising is None or surprisal_val > most_surprising[0]:
                most_surprising = (surprisal_val, words)

        # Post most surprising concept representation to the Global Workspace
        if most_surprising:
            self.last_concepts = self.lang.concepts(most_surprising[1])[:6]
            self.ws.post("text", self.last_concepts, min(1.0, most_surprising[0] / 10.0))

        return min(1.0, new_words / 10.0)

    def _digest_human(self) -> bool:
        """
        Consumes incoming human communicative messages from queue, learns words, and generates replies.

        Returns:
            bool: True if at least one human message was processed this cycle.
        """
        processed_any = False
        while True:
            try:
                msg = self.human_msgs.get_nowait()
            except queue.Empty:
                return processed_any

            processed_any = True
            # Ingest human message with boosted plasticity strength (4.0)
            self.lang.ingest(msg, strength=4.0)
            concept_tokens = self.lang.concepts(self.lang.words_of(msg))

            # Broadcast high-salience social item to workspace
            self.ws.post("human", concept_tokens[:4], 1.0)

            # Generate conversational response seeded with human concepts
            reply_seeds = concept_tokens[:3] or self.lang.words_of(msg)[:3]
            reply = self.lang.generate(reply_seeds, n=14, temp=0.5)
            self.replies.put(reply)

    def _pick_text(self) -> Optional[str]:
        """
        Selects a learned conceptual word to type into an input field.

        Returns:
            Optional[str]: Chosen word string, or None if no concepts are available.
        """
        workspace_winner = self.ws.winner()
        candidate_words = (workspace_winner["words"] if workspace_winner else []) or self.last_concepts
        return random.choice(candidate_words) if candidate_words else None

    # ---------------------------------------------------------------------------------------------
    # SLEEP CONSOLIDATION & CHECKPOINTING
    # ---------------------------------------------------------------------------------------------
    def sleep(self) -> None:
        """
        Executes biological sleep consolidation:
          1. Episodic Replay: Randomly samples past transitions from replay memory and updates policy.
          2. Synaptic Decay: Attenuates weak language associations to preserve memory capacity.
          3. Prototype Consolidation: Merges near-duplicate episodic prototype exemplars.
          4. Homeostatic Reset: Clears dead-action inhibition sets and resets fatigue to zero.
          5. Persistent Serialization: Writes state checkpoints to disk.
        """
        self.sleeps += 1
        replay_count = min(len(self.replay), 400)

        # 1. Offline episodic replay
        if replay_count > 0:
            sample_indices = self.rng.choice(len(self.replay), replay_count, replace=False)
            for idx in sample_indices:
                st, a, r, st_next = self.replay[int(idx)]
                self.policy.replay_update(st, a, r, st_next)

        # 2. Homeostatic language synaptic decay
        self.lang.decay(f=0.97, floor=0.05)

        # 3. Prototype engram consolidation
        merged_count = self.mem.consolidate()

        # 4. Inhibition and drive reset
        self.dead.clear()
        self.prev = None
        self.drives.rest()

        # 5. Save state to disk
        self.save()

        if self.verbose:
            print(f"   💤 sleep #{self.sleeps}: replayed {replay_count} episodes, merged {merged_count} scenes, "
                  f"vocab={len(self.lang.w2i)}, scenes={self.mem.n}, self={self.me.report()}")

    def save(self) -> None:
        """
        Atomically saves model weights, running statistics, episodic memory, and language cortex to disk.
        """
        os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)
        tmp_ckpt = self.ckpt + ".tmp.npz"
        np.savez_compressed(
            tmp_ckpt,
            encW=self.encoder.W,
            encB=self.encoder.bias,
            encR=self.encoder.rate,
            polW=self.policy.W,
            polV=self.policy.V,
            polU=self.policy.U,
            fwdM=self.fwd.M,
            memP=self.mem.P,
            memC=self.mem.count,
            memL=self.mem.last,
            memQ=self.mem.Qe,
            memN=np.array([self.mem.n]),
            retMu=self.retina.mu,
            retVar=self.retina.var,
            retN=np.array([self.retina.n]),
            t=np.array([self.t])
        )
        os.replace(tmp_ckpt, self.ckpt)
        self.lang.save(self.lang_path)

    def load(self, legacy_language: Optional[str] = None) -> None:
        """
        Loads network weights, running statistics, and language memories from disk.

        Args:
            legacy_language (Optional[str]): Path to legacy language NPZ file for backward compatibility.
        """
        if os.path.exists(self.ckpt):
            try:
                data = np.load(self.ckpt)
                if data["encW"].shape == self.encoder.W.shape and data["polW"].shape == self.policy.W.shape:
                    self.encoder.W = data["encW"]
                    self.encoder.bias = data["encB"]
                    self.encoder.rate = data["encR"]
                    self.policy.W = data["polW"]
                    self.policy.V = data["polV"]
                    self.policy.U = data["polU"]
                    self.fwd.M = data["fwdM"]
                    self.mem.P = data["memP"]
                    self.mem.count = data["memC"]
                    self.mem.last = data["memL"]
                    self.mem.Qe = data["memQ"]
                    self.mem.n = int(data["memN"][0])
                    self.retina.mu = data["retMu"]
                    self.retina.var = data["retVar"]
                    self.retina.n = int(data["retN"][0])
                    self.t = int(data["t"][0])
            except Exception as e:
                if self.verbose:
                    print(f"   (checkpoint not loaded: {e})")

        # Load language cortex associations
        if not self.lang.load(self.lang_path) and legacy_language:
            if self.lang.load(legacy_language) and self.verbose:
                print(f"   (imported language memory from {legacy_language})")

    # ---------------------------------------------------------------------------------------------
    # MAIN COGNITIVE RUN LOOP
    # ---------------------------------------------------------------------------------------------
    def run(self, max_steps: int = 0, delay: float = 0.0) -> None:
        """
        Executes continuous cognitive loop, advancing through perception, decision, action, and sleep.

        Args:
            max_steps (int): Total lifetime steps before stopping (0 for indefinite execution).
            delay (float): Minimum wall-clock delay in seconds per step to throttle execution rate.
        """
        self.running = True
        consecutive_errors = 0

        while self.running:
            step_start_time = time.time()
            try:
                record = self.step()
                consecutive_errors = 0

                if self.verbose:
                    print(self.format(record))

                # Flush vocal replies to stdout
                while not self.replies.empty():
                    print(f"   🗣️  {self.replies.get_nowait()}")

                # Check if circadian fatigue triggers sleep consolidation
                if self.drives.fatigue >= 1.0:
                    self.sleep()

            except Exception as e:
                consecutive_errors += 1
                print(f"   ⚠ step error ({consecutive_errors}): {e!r}")
                if consecutive_errors >= 5:
                    break

            if max_steps and self.t >= max_steps:
                break

            if delay > 0.0:
                elapsed = time.time() - step_start_time
                time.sleep(max(0.0, delay - elapsed))

        self.running = False
        self.save()

    @staticmethod
    def format(rec: Dict[str, Any]) -> str:
        """
        Formats a step telemetry record into an informative terminal log string.

        Args:
            rec (Dict[str, Any]): Telemetry dictionary from step().

        Returns:
            str: Single-line formatted console summary.
        """
        thought_str = f' 💭 "{rec["thought"]}"' if rec["thought"] else ""
        return (
            f"[{rec['t']:05d}] {action_name(rec['a']):<13} {str(rec['info'])[:18]:<18} "
            f"r={rec['r']:+.2f} nov={rec['nov']:.2f} chg={rec['change']:.2f} | "
            f"scenes={rec['pages']:3d} words={rec['vocab']:4d} | "
            f"cur={rec['curiosity']:.2f} fat={rec['fatigue']:.2f} conf={rec['conf']:.2f} "
            f"TD={rec['delta']:+.2f}{thought_str}"
        )


# =================================================================================================
# SYNTHETIC TEST ENVIRONMENT & VERIFICATION
# =================================================================================================
_MOCK_TEXT: List[str] = [
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
"""Corpus of multi-domain synthetic sentences for testing sensory text comprehension and language learning."""


class MockWeb:
    """
    Synthetic multi-page sensory web environment for simulating perception and navigation.

    Generates synthetic pages featuring distinct visual textures, vertical stripe structures,
    background colors, interactive blue link affordances, and natural language text.

    Attributes:
        rng (np.random.RandomState): Seeded pseudo-random number generator.
        n_pages (int): Total number of synthetic web pages in the simulation.
        pages (List[Dict[str, Any]]): List of generated page dictionaries.
        cur (int): Current active page index.
        hist (List[int]): Browser navigation history stack for backstep actions.
        scroll (int): Current vertical scroll position (0: top, 1: middle, 2: bottom).
        visited (set): Set of unique page indices visited by the agent.
        prev_lum (Optional[np.ndarray]): Cached luminance frame from preceding step for motion calculation.
        _pending (Optional[str]): Pending text buffer waiting to be read via poll_text().
        supports_typing (bool): Environmental capability flag indicating typing actions are unsupported.
        clicks (int): Cumulative count of click actions executed.
        effective (int): Cumulative count of clicks that successfully navigated to a new page.
    """

    def __init__(self, n_pages: int = 30, seed: int = 0) -> None:
        """
        Initializes the synthetic web environment with n_pages interconnected pages.

        Args:
            n_pages (int): Total number of procedural pages to generate (default: 30).
            seed (int): Random seed for procedural generation.
        """
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self.n_pages: int = n_pages
        self.pages: List[Dict[str, Any]] = []

        for p in range(n_pages):
            # 1. Procedural background texture: 9x16 grid scaled by Kronecker product up to 36x64
            tex = np.kron(self.rng.rand(9, 16, 3), np.ones((4, 4, 1)))

            # 2. Vertical layout stripes: ~40% of grid rows receive a stripe
            stripes = (self.rng.rand(GRID_H, 1) > 0.6).astype(np.float32)

            # 3. Base background RGB tint
            base = self.rng.rand(3) * 0.7 + 0.1

            # 4. Interactive clickable links: 3 unique cells chosen from the 32 receptive fields
            links = list(self.rng.choice(N_CELLS, 3, replace=False))

            # 5. Destination targets for each link (ensuring target != current page)
            other_pages = [q for q in range(n_pages) if q != p]
            targets = [int(self.rng.choice(other_pages)) for _ in links]

            # 6. Package page attributes
            page_text = _MOCK_TEXT[p % len(_MOCK_TEXT)] + f" number {p}."
            self.pages.append({
                "tex": tex,
                "stripes": stripes,
                "base": base,
                "links": links,
                "targets": targets,
                "text": page_text
            })

        # Browser state
        self.cur: int = 0
        self.hist: List[int] = []
        self.scroll: int = 0
        self.visited: set = {0}
        self.prev_lum: Optional[np.ndarray] = None
        self._pending: Optional[str] = None
        self.supports_typing: bool = False
        self.clicks: int = 0
        self.effective: int = 0

    def _render(self) -> np.ndarray:
        """
        Renders the current visual state into a 64x36 RGB pixel image.

        Incorporates page background, texture, vertical stripes modulated by scroll offset,
        and high-contrast blue rectangular link affordances.

        Returns:
            np.ndarray: Rendered RGB visual frame of shape (GRID_H, GRID_W, 3) in [0.0, 1.0], float32.
        """
        page = self.pages[self.cur]

        # Combine base color and scaled texture
        img = page["base"][None, None, :] + 0.15 * (page["tex"] - 0.5)

        # Apply scrolling vertical stripes
        shifted_stripes = np.roll(page["stripes"], 5 * self.scroll, axis=0)[:, :, None]
        img = img + 0.25 * shifted_stripes * np.ones(3)

        # Draw blue clickable link rectangles at corresponding fixation cell coordinates
        for link_cell in page["links"]:
            col = link_cell % CELL_COLS
            row = link_cell // CELL_COLS
            y0 = row * (GRID_H // CELL_ROWS)
            x0 = col * (GRID_W // CELL_COLS)
            # Outer dark blue button body
            img[y0:y0 + 9, x0:x0 + 8] = (0.10, 0.20, 0.85)
            # Inner light blue highlight pattern
            img[y0 + 2:y0 + 9:3, x0 + 1:x0 + 7] = (0.45, 0.55, 0.95)

        # Add slight sensor noise to simulate camera/rendering noise and clip to [0, 1]
        noise = self.rng.randn(GRID_H, GRID_W, 3) * 0.01
        return np.clip(img + noise, 0.0, 1.0).astype(np.float32)

    def observe(self) -> Dict[str, Any]:
        """
        Captures current observation package comprising rendered RGB frame and motion flux map.

        Returns:
            Dict[str, Any]: Observation dictionary containing 'rgb' and 'motion'.
        """
        rgb = self._render()
        lum = Retina.lum_of(rgb)
        motion = np.abs(lum - self.prev_lum) if self.prev_lum is not None else np.zeros_like(lum)
        self.prev_lum = lum
        return {"rgb": rgb, "motion": motion.astype(np.float32)}

    def request_read(self) -> None:
        """Stages the natural text of the current page into the pending buffer."""
        self._pending = self.pages[self.cur]["text"]

    def poll_text(self) -> Optional[str]:
        """
        Retrieves and clears the pending text buffer.

        Returns:
            Optional[str]: Text string if available, or None.
        """
        text = self._pending
        self._pending = None
        return text

    def act(self, a: int, text: Optional[str] = None) -> str:
        """
        Executes a discrete motor action within the synthetic web environment.

        Args:
            a (int): Motor action index.
            text (Optional[str]): Text parameter for typing actions (unused in MockWeb).

        Returns:
            str: Telemetry feedback string describing action outcome.
        """
        page = self.pages[self.cur]

        if a < N_CELLS:
            self.clicks += 1
            # If clicked cell contains an active link, navigate with 95% physical reliability
            if a in page["links"] and self.rng.rand() > 0.05:
                link_index = page["links"].index(a)
                target_page = page["targets"][link_index]
                self.hist.append(self.cur)
                self.cur = target_page
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
    """
    Executes an isolated simulation trial on the MockWeb environment.

    Args:
        seed (int): Random seed for trial environment and agent initialization.
        steps (int): Total cognitive execution steps to run.
        explore_only (bool): If True, runs an unguided random explorer as an experimental control baseline.

    Returns:
        Dict[str, Any]: Trial metrics including visited page count, late effective click rate, vocab size, etc.
    """
    env = MockWeb(n_pages=30, seed=seed)
    temp_dir = tempfile.mkdtemp()
    org = Organism(
        env,
        ckpt_dir=temp_dir,
        seed=seed,
        explore_only=explore_only,
        verbose=False,
        autosave_every=0,
        sleep_every=300
    )

    halfway_step = steps // 2
    clicks_at_half = [0, 0]
    reached_at: Optional[int] = None

    for i in range(steps):
        org.step()
        if org.drives.fatigue >= 1.0:
            org.sleep()
        if i == halfway_step:
            clicks_at_half = [env.clicks, env.effective]
        if reached_at is None and len(env.visited) >= 20:
            reached_at = i

    late_clicks = max(1, env.clicks - clicks_at_half[0])
    late_effective = env.effective - clicks_at_half[1]

    return {
        "visited": len(env.visited),
        "late_effective_rate": late_effective / late_clicks,
        "reached20": reached_at if reached_at is not None else steps,
        "vocab": len(org.lang.w2i),
        "scenes": org.mem.n,
        "org": org
    }


def selftest() -> bool:
    """
    Comprehensive verification suite testing all core cognitive dynamics:
      1. Learning vs. Random Baseline (5 seeds x 1500 steps).
      2. Receptive Field Affordance Generalization.
      3. Language Acquisition and Surprisal Habituation.
      4. Inner Speech Monologue Generation.
      5. State Persistence Round-Trip Checkpoint Serialization.
      6. Sleep Consolidation (episodic replay and synaptic decay).

    Returns:
        bool: True if all cognitive verification tests pass, False otherwise.
    """
    ok_all = True

    def check(name: str, ok: bool, extra: str = "") -> None:
        nonlocal ok_all
        ok_all &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} {extra}")

    print("== Learning vs. Random Baseline on Synthetic Web (5 seeds x 1500 steps) ==")
    learners: List[Dict[str, Any]] = []
    randoms: List[Dict[str, Any]] = []

    for s in range(5):
        learners.append(_run_mock(seed=s, steps=1500, explore_only=False))
        randoms.append(_run_mock(seed=s, steps=1500, explore_only=True))

    calc_mean = lambda results, metric_key: float(np.mean([r[metric_key] for r in results]))

    print(f"  {'':22}{'learner':>10}{'random':>10}")
    metrics = [
        ("late_effective_rate", "effective-click rate (2nd half)"),
        ("visited", "pages visited (of 30)"),
        ("reached20", "steps to visit 20 pages")
    ]
    for key, label in metrics:
        print(f"  {label:<32}{calc_mean(learners, key):>10.3f}{calc_mean(randoms, key):>10.3f}")

    # Assertions on learning dynamics
    learner_eff = calc_mean(learners, "late_effective_rate")
    random_eff = calc_mean(randoms, "late_effective_rate")
    check(
        "learner clicks links far more often than chance",
        learner_eff > 2.0 * random_eff,
        f"({learner_eff:.2f} vs {random_eff:.2f})"
    )

    learner_reached = calc_mean(learners, "reached20")
    random_reached = calc_mean(randoms, "reached20")
    check(
        "learner covers the web faster (mean steps to 20 pages)",
        learner_reached < 0.6 * random_reached,
        f"({learner_reached:.0f} vs {random_reached:.0f})"
    )

    print("\n== Language / Sleep / Persistence ==")
    test_org = learners[0]["org"]

    # Vocabulary acquisition
    vocab_size = len(test_org.lang.w2i)
    check("learned vocabulary from sensory text", vocab_size > 40, f"(vocab={vocab_size})")

    # Surprisal habituation: seen text should be significantly less surprising than unseen text
    surprisal_seen = test_org.lang.surprisal(test_org.lang.words_of(_MOCK_TEXT[0]))
    surprisal_novel = test_org.lang.surprisal(test_org.lang.words_of("quantum chromodynamics describes gluon exchange"))
    check(
        "habituation: seen text less surprising than unseen text",
        surprisal_seen < surprisal_novel,
        f"({surprisal_seen:.1f} < {surprisal_novel:.1f} bits)"
    )

    # Inner speech generation
    thought = test_org.lang.generate(["cortex"], n=8)
    check("inner speech generates from learned associations", len(thought.split()) > 1, f'("{thought}")')

    # Persistence round-trip
    test_org.save()
    reloaded_org = Organism(MockWeb(seed=0), ckpt_dir=os.path.dirname(test_org.ckpt), verbose=False, autosave_every=0)
    check(
        "checkpoint round-trip",
        reloaded_org.mem.n == test_org.mem.n and len(reloaded_org.lang.w2i) == len(test_org.lang.w2i)
    )

    # Sleep consolidation (replay + decay)
    bigrams_before = len(test_org.lang.bi)
    test_org.sleep()
    check(
        "sleep consolidation (replay + decay)",
        test_org.sleeps >= 1 and len(test_org.lang.bi) <= bigrams_before
    )

    print("\nSELF-TEST", "PASSED" if ok_all else "FAILED")
    return ok_all


def main() -> None:
    """Command-line entrypoint for running the Organism core cognitive model."""
    parser = argparse.ArgumentParser(description="Organism: Core Cognitive Model (strictly zero backpropagation)")
    parser.add_argument("--selftest", action="store_true", help="run comprehensive verification suite")
    parser.add_argument("--steps", type=int, default=100, help="run mock learning steps (default: 100)")
    parser.add_argument("--sleep-every", type=int, default=300, help="interval in steps between sleep consolidation")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    print("Running Organism core model in synthetic environment...")
    env = MockWeb(n_pages=30, seed=42)
    org = Organism(env, sleep_every=args.sleep_every)
    org.run(max_steps=args.steps)
    print(f"\nDone. Steps={org.t}, Scenes={org.mem.n}, Vocab={len(org.lang.w2i)}, Stats={org.me.report()}")


if __name__ == "__main__":
    main()