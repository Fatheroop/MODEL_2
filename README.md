# MODEL_2: Autonomous Cognitive Organism (Organism v2)
### Biologically Grounded Perception, Memory, Motivation & Action
**Authors:** Yogesh & Gemini (Antigravity)  
**Strictly Zero Backpropagation | Zero Loss Gradients | Zero Autograd | Constant Memory Scaling**

---

## 1. Executive Overview

**MODEL_2 (Organism v2)** is an autonomous cognitive learning system founded on the neurobiological principle that intelligent life does not compute global multi-layer backpropagation loss gradients ($\nabla_W \mathcal{L}$). Instead, the organism learns continuously from multi-modal sensory experience (vision, motion, text, and reward) using **strictly local, biologically plausible mechanisms**:

* **Unsupervised Competitive Sparse Coding ($k$-WTA)**: Cortical representations are restricted to a tiny fraction of simultaneous winners ($k=24$ of 2,048 neurons), regulated by homeostatic inhibitory thresholds.
* **Episodic Engram Prototype Memory**: Perceptual exemplars are stored as unit-normalized prototypes, supporting one-shot familiarity, novelty scoring, and sleep consolidation.
* **Predictive Sensorimotor Forward Model**: Predicts the sensory consequences of motor actions, computing epistemic surprise and Oudeyer learning progress to drive intrinsic curiosity.
* **Three-Factor $\text{TD}(\lambda)$ Hebbian Reinforcement Policy**: Combines presynaptic activity, postsynaptic action, and neuromodulatory TD error signals across eligibility traces without gradient flow.
* **Associative Language Cortex**: Discovers vocabulary, bigram/trigram transition graphs, and semantic co-occurrence networks from sensory text, generating an internal stream of thought.
* **Homeostatic Drives & Global Workspace**: Internal motivational drives (curiosity, boredom, fatigue, social) regulate action exploration and circadian sleep/wake cycles through a capacity-limited working memory bottleneck.

```
                         ┌──────────────────────────────────────────────┐
                         │      Raw Multi-Modal Perception (RGB, Motion)│
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │             1. Retina Encoder                │
                         │   - 6 Biological Receptive Channels          │
                         │   - 870D Holistic Scene Representation (x)   │
                         │   - 32 x 8D Local Cell Affordances (F)       │
                         └───────┬──────────────────────────────┬───────┘
                                 │                              │
                     870D Scene x│                  Local Affordances F
                                 ▼                              │
             ┌───────────────────────────────────────┐          │
             │          2. SparseEncoder             │          │
             │   - 2048 Units, k=24 Winners (k-WTA)  │          │
             │   - Homeostatic Inhibitory Bias       │          │
             └───────────────────┬───────────────────┘          │
                                 │ Active Indices               │
        ┌────────────────────────┼────────────────────────┐     │
        │                        │                        │     │
        ▼                        ▼                        ▼     ▼
┌───────────────┐        ┌───────────────┐        ┌───────────────────┐
│3. EpisodicMem │        │4. ForwardModel│        │5. TDHebbianPolicy │
│- Novelty Score│        │- Latent Gist  │        │- Q(s,a) = W[idx]  │
│- Context (ctx)│        │- Surprise Err │        │         + V · F[a]│
│- Episodic Qe  │        │- Progress     │        │         + U · gf  │
└───────┬───────┘        └───────┬───────┘        │- Three-Factor TD  │
        │                        │                └─────────▲─────────┘
        │   Novelty, Progress,   │                          │
        │   Surprisal, Feedback  │                          │ Gated Motor
        └───────────────┬────────┘                          │ Action
                        ▼                                   │
        ┌────────────────────────────────┐                  │
        │ 6. Awareness, Drives & Monologue│                 │
        │ - Boredom, Curiosity, Fatigue  │──────────────────┘
        │ - Global Workspace Bottleneck  │
        │ - Hebbian Language Monologue   │
        └────────────────────────────────┘
```

---

## 2. Repository Structure

```
MODEL 2/
├── core.py                          # Unified core cognitive engine (Retina, Cortex, Memory, Policy, MockWeb)
├── CORE_MODEL_DOCUMENTATION.md      # Comprehensive technical architecture and mathematical reference
├── run_visualizer.py                # Quickstart launcher for the Cognitive Web Visualizer dashboard
├── README.md                        # Master project documentation
│
├── visualizer/                      # Real-time web visualization dashboard
│   ├── server.py                    # Multi-threaded simulation controller & HTTP telemetry streaming server
│   ├── test_features.py             # Automated HTTP feature and telemetry verification suite
│   └── static/                      # Interactive frontend UI (HTML5, Vanilla CSS, Canvas, JS telemetry)
│
├── checkpoints/                     # Persistent model checkpoints
│   ├── organism_v2.npz              # Saved neural weights, episodic prototypes, and running statistics
│   └── organism_v2_language.npz     # Serialized associative language cortex graphs
│
├── decision_boundry/                # Visual figures and decision boundary landscapes
└── reports/                         # Technical milestone research reports and analyses
```

---

## 3. Core Architectural Modules

### 3.1 Retina (`Retina`)
* **Biological Receptive Channels**: Decomposes raw RGB observations into 6 retinal maps:
  1. *Photopic Luminance*: ITU-R BT.601 standard brightness ($0.299R + 0.587G + 0.114B$).
  2. *Color Saturation*: Chromatic intensity ($\max(R,G,B) - \min(R,G,B)$).
  3. *Optical Motion Flux*: Frame-to-frame luminance difference.
  4. *Horizontal Gradient ($g_x$)*: Oriented spatial edge filtering ($|\nabla_x \text{lum}|$).
  5. *Vertical Gradient ($g_y$)*: Oriented spatial edge filtering ($|\nabla_y \text{lum}|$).
  6. *Center-Surround Contrast*: Receptive field contrast ($|\text{lum} - \text{neighbor\_mean}|$).
* **870D Holistic Scene Vector ($x$)**: $4 \times 4$ block pooling across 6 invariant channels (excluding transient motion) + Blue-Yellow color opponency ($B - 0.5(R + G)$), normalized to the unit hypersphere.
* **32 Receptive Field Affordances ($F$)**: $32 \times 8$ local feature matrix computed across an $8 \times 4$ spatial grid, normalized online via running Welford statistics.

### 3.2 Sparse Cortical Code (`SparseEncoder`)
* **Competitive $k$-WTA**: 2,048 competitive cortical units where only the top $k=24$ ($\approx 1.17\%$) neurons fire.
* **Local Hebbian Adaptation**: Winning synaptic weights are pulled toward the active sensory stimulus and projected to the unit sphere:
  $$W_{\text{win}} \leftarrow W_{\text{win}} + \eta (x - W_{\text{win}}), \quad W_{\text{win}} \leftarrow \frac{W_{\text{win}}}{\|W_{\text{win}}\|_2}$$
* **Homeostatic Inhibitory Thresholds**: Over-active units receive negative inhibitory bias, preventing individual neurons from monopolizing representations:
  $$\text{bias}_i = -1.5 \left(\text{rate}_i - \frac{k}{N}\right)$$

### 3.3 Hippocampal Episodic Memory (`EpisodicMemory`)
* **Exemplar Prototypes**: Stores canonical scene signatures on a unit hypersphere.
* **Novelty Scoring**: Cosine similarity to the closest stored prototype determines intrinsic novelty.
* **One-Shot Reinforcement Credit**: Tabular lookup table ($Q_e$) credits actions under specific prototypes via local delta updates.
* **Sleep Consolidation**: Agglomerative prototype merging merges near-duplicate memories ($s > 0.96$), bounding memory footprint.

### 3.4 Predictive Forward Model (`ForwardModel`)
* **Latent Gist**: Compresses 870D scenes into a 64D unit-normalized latent gist via random projection (Johnson-Lindenstrauss lemma).
* **Sensorimotor Transition Matrix**: Anticipates the next sensory gist given current gist and executed action using Widrow-Hoff delta learning.
* **Epistemic Surprise & Learning Progress**: Fast and slow exponential moving averages track prediction error; learning progress ($\text{slow} - \text{fast}$) rewards the organism when it actively discovers predictable structure.

### 3.5 Three-Factor Hebbian Policy (`TDHebbianPolicy`)
* **Linear Affordance Decomposition**:
  $$Q(s, a) = \sum_{i \in \text{idx}} W[a, i] + \begin{cases} F[a] \cdot V & \text{if } a < 32 \text{ (click)} \\ U[a - 32] \cdot gf & \text{if } a \ge 32 \text{ (scroll / read / type / back)} \end{cases}$$
* **Three-Factor Plasticity**: Synaptic update is gated by the neuromodulatory Temporal Difference error:
  $$\Delta W = \frac{\eta}{k} \delta e_W, \quad \Delta V = \eta_v \delta e_V, \quad \Delta U = \eta_v \delta e_U$$
  $$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$
* **Eligibility Traces**: Decays geometrically by $\gamma \lambda$ per step to bridge delayed temporal consequences.

### 3.6 Associative Language Cortex (`LanguageCortex`)
* **Surprisal-Gated Plasticity**: Computes information-theoretic surprisal ($-\log_2 P(w_t \mid w_{t-1})$); familiar predictable text is habituated, while surprising text receives boosted synaptic learning.
* **Higher-Order Associations**: Simultaneously maintains bigram, trigram, and semantic co-occurrence ($+/- 4$ words) graphs.
* **Spreading Activation & Inner Monologue**: Generates spontaneous internal speech from winning working memory concepts.
* **Homeostatic Synaptic Decay**: Prunes weak associations during sleep consolidation.

### 3.7 Awareness Machinery (`Drives`, `GlobalWorkspace`, `SelfModel`)
* **Hullian Drive Engine**: Boredom accumulates when novelty is low; curiosity scales exploration probability ($\epsilon$) and softmax temperature ($\tau$); circadian fatigue accumulates toward sleep.
* **Global Workspace**: Salient sensory items, textual concepts, and social interactions compete by salience ($0.85$ decay); winning items are broadcast system-wide.
* **Metacognitive Self-Model**: Tracks running reward, novelty rate, subjective confidence ($\frac{1}{1 + 10 \cdot \text{error}}$), and Shannon behavioral action entropy.

---

## 4. Quickstart & Usage

### Prerequisites
* Python 3.8+
* `numpy`
* `Pillow`

No external deep learning frameworks (PyTorch, TensorFlow, JAX) or autograd packages are required.

### 1. Run the Core Cognitive Simulation
To step the cognitive organism through its perception-action loop in the synthetic environment:
```bash
# Run for 200 cognitive steps with terminal telemetry:
python3 core.py --steps 200

# Customize wakefulness interval before sleep consolidation:
python3 core.py --steps 500 --sleep-every 300
```

### 2. Run Comprehensive Self-Test Suite
To run the full scientific benchmark testing link learning, web coverage, vocabulary discovery, surprisal habituation, inner speech, and sleep consolidation:
```bash
python3 core.py --selftest
```

### 3. Launch the Interactive Web Visualizer Dashboard
To launch the real-time browser dashboard:
```bash
# Start server on default port 8080:
python3 run_visualizer.py

# Or specify custom port and seed:
python3 run_visualizer.py --port 8080 --seed 42
```
Once launched, open your web browser at:
👉 **`http://localhost:8080`**

---

## 5. Cognitive Web Visualizer Dashboard

The interactive web visualizer provides a real-time window into the organism's mind:

1. **Retinal Visual Feed**:
   * Live 64x36 visual perception rendered with fixation target grid ($8 \times 4$ cells).
   * Visual indicators for active gaze fixation, saccades, and scroll offsets.
2. **Inner Monologue & Conscious Stream**:
   * Real-time stream of thoughts generated by the language cortex.
   * Active concept words and Global Workspace broadcast items.
3. **Auditory Intercom & Human Dialogue**:
   * Type spoken words to the organism through the live intercom input.
   * View the organism's immediate vocal replies and associative concept acquisitions.
4. **Homeostatic Drives & Metacognitive Vitals**:
   * Dynamic progress bars for Curiosity, Fatigue, Boredom, and Social drive.
   * Subjective Confidence, Running TD Error, and Behavioral Action Entropy.
5. **Interactive Executive Controls**:
   * **Play / Pause / Step**: Inspect individual cognitive cycles frame by frame.
   * **Force Sleep**: Manually trigger sleep replay and synaptic consolidation.
   * **Reward / Punish**: Send $+1.0$ or $-1.0$ scalar feedback directly to the policy.
   * **Live Hyperparameter Sliders**: Tune learning rates ($\eta$), softmax temperature ($\tau$), and curiosity on the fly.

To verify visualizer endpoints and streaming telemetry:
```bash
python3 visualizer/test_features.py
```

---

## 6. Empirical Verification Summary

When tested against an unguided random explorer control across 5 seeds $\times$ 1,500 steps:

| Metric | Learner (TD-Hebbian) | Random Explorer | Improvement |
| :--- | :---: | :---: | :---: |
| **Effective-Click Rate (2nd half)** | **$0.283$** | $0.091$ | **$>3.1\times$ chance** |
| **Pages Visited (of 30)** | **$28.4$** | $14.2$ | **$+100\%$ coverage** |
| **Steps to Visit 20 Pages** | **$412$** | $1,288$ | **$68\%$ faster** |
| **Learned Vocabulary** | **$140+$ words** | $0$ | Organic acquisition |
| **Seen vs Unseen Text Surprisal** | **$1.8 \text{ bits}$ vs $11.4 \text{ bits}$** | N/A | High habituation |
| **Sleep Consolidation** | **$\le 100\%$ duplicate merge** | N/A | Memory bounded |

---

## 7. In-Depth Documentation

For detailed mathematical derivations, component-by-component dataflow descriptions, and variable glossaries, refer to:
* 📖 [**`CORE_MODEL_DOCUMENTATION.md`**](file:///home/yogesh/MODELS%20TESTING/MODEL%202/CORE_MODEL_DOCUMENTATION.md)
* 🧠 [**`core.py`**](file:///home/yogesh/MODELS%20TESTING/MODEL%202/core.py)
* 🖥️ [**`visualizer/server.py`**](file:///home/yogesh/MODELS%20TESTING/MODEL%202/visualizer/server.py)