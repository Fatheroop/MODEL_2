# Core Cognitive Model (Organism v2)
## Complete Technical Architecture & Codebase Documentation
**Zero Backpropagation | Zero Loss Gradients | Zero Autograd | Constant Memory Scaling**

---

## 1. Architectural Overview & Philosophy

The **Organism Core Cognitive Model** is an autonomous perception-cognition-action agent grounded in mammalian neurobiology. Rather than optimizing millions of distributed weights via artificial multi-layer backpropagation of errors ($E = \frac{1}{2}(y - \hat{y})^2$ through $\nabla_W E$), this system operates strictly on **local, biologically plausible mechanisms**:

1. **Local Synaptic Plasticity**:
   Synaptic updates depend strictly on local pre-synaptic firing, post-synaptic activation, and diffuse neuromodulatory signals (dopaminergic Temporal Difference errors or noradrenergic novelty signals).
2. **Competitive Sparse Coding (k-WTA)**:
   Cortical minicolumn representations are restricted to a tiny fraction ($k \ll N$, e.g., 24 of 2,048 neurons) of simultaneous winners, maintaining high representational capacity with minimal energy and interference.
3. **Episodic Exemplar Memorization**:
   Unfamiliar scenes are stored directly as discrete engram prototypes in an episodic memory bank, enabling one-shot recognition, novelty quantification, and rapid contextual credit assignment.
4. **Predictive Processing & Intrinsic Curiosity**:
   The organism continuously predicts the sensory consequences of its own motor actions. Prediction errors trigger epistemic surprise; falling prediction error generates *learning progress*, rewarding the organism for exploring learnable environments while ignoring unlearnable noise.
5. **Associative Lexical & Semantic Graphs**:
   Language is acquired incrementally from sensory text into N-gram transition matrices and semantic co-occurrence networks, gated by information-theoretic surprisal, and voiced through an internal monologue (inner speech).
6. **Circadian Sleep Consolidation**:
   Periodic sleep phases trigger offline episodic replay, homeostatic synaptic forgetting, and prototype agglomeration, stabilizing knowledge while bounding memory footprint.

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

## 2. Geometry, Resolution & Action Space

### Visual Field Constants
* `GRID_W = 64`: Horizontal visual resolution in pixels.
* `GRID_H = 36`: Vertical visual resolution in pixels.
* Aspect ratio is 16:9 widescreen, matching modern visual displays.

### Receptive Field Fixation Grid
* `CELL_COLS = 8`: 8 horizontal columns of fixation targets.
* `CELL_ROWS = 4`: 4 vertical rows of fixation targets.
* `N_CELLS = 32`: Total spatial fixation cells ($8 \times 4 = 32$).
* Each receptive field covers a spatial patch of $8 \times 9$ pixels ($64/8 \times 36/4$).

### Motor Action Space
* Total Discrete Actions: `N_ACTIONS = 37` (`N_CELLS + 5`).
* Actions `0 .. 31`: Fixation click at grid cell coordinates `(col, row) = (a % 8, a // 8)`.
* Action `32` (`A_SCROLL_DOWN`): Scroll down content viewport.
* Action `33` (`A_SCROLL_UP`): Scroll up content viewport.
* Action `34` (`A_READ`): Dwell gaze on screen to parse textual content.
* Action `35` (`A_TYPE`): Select a salient concept from working memory and type it.
* Action `36` (`A_BACK`): Navigate back to the previous screen.

### Vector Dimensionality
* `N_LOC = 8`: Dimensionality of local visual affordance vectors extracted for each of the 32 click cells.
* `X_DIM = 870`: Dimensionality of the pooled holistic scene vector.
  * 6 retinal channels spatially downsampled via $4 \times 4$ block averaging into a $9 \times 16 = 144$ block grid:
    $$6 \times 144 = 864 \text{ regional features}$$
  * 6 global channel mean intensities appended:
    $$864 + 6 = 870 \text{ dimensions}$$

---

## 3. Subsystem Breakdown: Classes, Methods & Variables

### 3.1 Class: `Retina`
Biological receptive field visual encoder modeling photopic luminance, color opponency, motion flux, and edge energy.

#### Instance Variables & Attributes
* `mu` (`np.ndarray`, shape: `(8,)`, `float32`): Online running mean vector of local cell affordances.
* `var` (`np.ndarray`, shape: `(8,)`, `float32`): Online running variance vector of local cell affordances.
* `n` (`int`): Count of processed visual frames used for Welford/exponential statistic updates.

#### Methods
* `lum_of(rgb: np.ndarray) -> np.ndarray`:
  * **Equation**: $\text{Lum} = 0.299 R + 0.587 G + 0.114 B$ (ITU-R BT.601 standard photopic luminance).
  * **Role**: Converts 3-channel color tensors to perceptual brightness.
* `encode(rgb: np.ndarray, motion: np.ndarray) -> np.ndarray`:
  * **Channels**:
    1. Luminance ($\text{lum}$)
    2. Saturation ($\max(R, G, B) - \min(R, G, B)$)
    3. Motion Flux ($\text{clip}(\text{motion}, 0, 1)$)
    4. Horizontal Gradient ($|\nabla_x \text{lum}|$)
    5. Vertical Gradient ($|\nabla_y \text{lum}|$)
    6. Center-Surround Contrast ($|\text{lum} - \text{neighbor\_avg}|$)
  * **Output**: Map of shape `(36, 64, 6)`.
* `pooled(m6: np.ndarray, rgb: np.ndarray) -> np.ndarray`:
  * **Role**: Computes scale-invariant, unit-normalized 870D holistic scene vector $x$.
  * **Note**: Motion channel is deliberately omitted from $x$ because transient motion does not constitute scene identity.
* `cell_features(m6: np.ndarray, rgb: np.ndarray) -> np.ndarray`:
  * **Role**: Computes $32 \times 8$ local affordance matrix $F$.
  * For each cell: 6 channel spatial means + luminance standard deviation + Blue-Yellow color opponency ($B - 0.5(R + G)$).
  * Online standardization: $F = \text{clip}\left(\frac{\text{raw} - \mu}{\sqrt{\sigma^2 + 10^{-4}}}, -3.0, 3.0\right) \times 0.5$.

---

### 3.2 Class: `SparseEncoder`
Cortical sparse coding layer driven by competitive k-Winners-Take-All (k-WTA) Hebbian dynamics with homeostatic threshold regulation.

#### Instance Variables & Attributes
* `W` (`np.ndarray`, shape: `(2048, 870)`, `float32`): Synaptic receptive field matrix. Each row is normalized to unit norm ($\|W_i\|_2 = 1.0$).
* `bias` (`np.ndarray`, shape: `(2048,)`, `float32`): Dynamic homeostatic inhibitory threshold offsets.
* `rate` (`np.ndarray`, shape: `(2048,)`, `float32`): Exponential moving average firing rate of each cortical unit.
* `n_units` (`int = 2048`): Total number of competitive cortical units.
* `k` (`int = 24`): Sparsity constraint; exactly 24 units fire per visual saccade ($\sim 1.17\%$ active).
* `eta` (`float = 0.05`): Synaptic plasticity rate.

#### Methods
* `encode(x: np.ndarray, learn: bool = True) -> np.ndarray`:
  * **Membrane Potential**: $u = W x + \text{bias}$.
  * **k-WTA Selection**: $\text{idx} = \text{top\_k}(u, k=24)$.
  * **Hebbian Plasticity** (for winners $i \in \text{idx}$):
    $$W_i \leftarrow W_i + \eta (x - W_i), \quad W_i \leftarrow \frac{W_i}{\|W_i\|_2}$$
  * **Homeostatic Inhibitory Bias**:
    $$\text{rate}_i \leftarrow \text{rate}_i + 0.002 (\mathbf{1}_{i \in \text{idx}} - \text{rate}_i)$$
    $$\text{bias}_i = -1.5 \left(\text{rate}_i - \frac{k}{N}\right)$$
    *If a neuron fires too often ($\text{rate} > k/N$), bias becomes negative, raising its firing threshold.*

---

### 3.3 Class: `EpisodicMemory`
Hippocampal-style non-parametric exemplar memory for prototype clustering, novelty scoring, and one-shot reinforcement credit assignment.

#### Instance Variables & Attributes
* `P` (`np.ndarray`, shape: `(capacity, 870)`, `float32`): Matrix of stored prototype scene vectors.
* `count` (`np.ndarray`, shape: `(capacity,)`, `float32`): Visit frequency tally for each prototype.
* `last` (`np.ndarray`, shape: `(capacity,)`, `float32`): Timestep of the most recent encounter.
* `Qe` (`np.ndarray`, shape: `(capacity, 37)`, `float32`): Cached episodic action values per prototype.
* `n` (`int`): Current number of allocated prototypes in memory.
* `cap` (`int = 600`): Maximum prototype capacity before eviction.
* `alloc` (`float = 0.30`): Cosine distance allocation threshold.

#### Methods
* `observe(x: np.ndarray, step: int) -> Tuple[int, float, float]`:
  * Computes cosine similarity: $s_i = P_i \cdot x$.
  * Identifies nearest prototype: $j = \text{argmax}_i s_i$.
  * **Novelty**: $\text{nov} = \text{clip}\left(\frac{1.0 - s_j - 0.10}{0.5}, 0.0, 1.0\right)$.
  * **Allocation**: If $1.0 - s_j > \text{alloc}$, registers new prototype slot (evicts least-used if full). Otherwise, refines existing prototype: $P_j \leftarrow \text{norm}(P_j + 0.1(x - s_j P_j))$.
* `credit(j: int, a: int, r: float)`:
  * Delta-rule credit assignment: $Q_e[j, a] \leftarrow Q_e[j, a] + 0.3(r - Q_e[j, a])$.
* `consolidate(thresh: float = 0.96) -> int`:
  * Merges near-duplicate prototypes during sleep whose cosine similarity exceeds `0.96`.

---

### 3.4 Class: `ForwardModel`
Sensorimotor predictive forward model anticipating future visual gist and quantifying learning progress.

#### Instance Variables & Attributes
* `R` (`np.ndarray`, shape: `(64, 870)`, `float32`): Random projection matrix (Johnson-Lindenstrauss compression).
* `M` (`np.ndarray`, shape: `(64, 101)`, `float32`): Predictive transition matrix mapping $[g_t, \text{one\_hot}(a_t)]$ to $g_{t+1}$.
* `fast` (`float = 0.15`): Fast EMA of prediction error ($\alpha = 0.30$).
* `slow` (`float = 0.15`): Slow EMA of prediction error ($\alpha = 0.02$).

#### Methods & Properties
* `gist(x: np.ndarray) -> np.ndarray`: Compresses 870D scene $x$ to 64D unit-normalized gist $g = \text{norm}(R x)$.
* `update(g: np.ndarray, a: int, g_next: np.ndarray, eta: float = 0.03) -> float`:
  * Input vector: $z = [g, \text{one\_hot}(a)]$.
  * Error vector: $e = g_{\text{next}} - M z$.
  * Local delta update: $M \leftarrow M + \eta (e \otimes z)$.
  * Fast/Slow error updates: $\text{fast} \leftarrow \text{fast} + 0.3 (|e| - \text{fast})$, $\text{slow} \leftarrow \text{slow} + 0.02 (|e| - \text{slow})$.
* `progress` (property): Returns $\text{slow} - \text{fast}$. Positive value denotes error reduction (active comprehension).

---

### 3.5 Class: `TDHebbianPolicy`
Three-Factor TD(lambda) reinforcement learning policy without backpropagation.

#### Instance Variables & Attributes
* `W` (`np.ndarray`, shape: `(37, 2048)`, `float32`): Cortical unit-to-action synaptic weights.
* `V` (`np.ndarray`, shape: `(8,)`, `float32`): Shared local visual affordance filter for click cells.
* `U` (`np.ndarray`, shape: `(5, 8)`, `float32`): Global affordance filters for non-click actions (scroll, read, type, back).
* `eW, eV, eU`: Eligibility trace accumulators decaying by $\gamma \lambda$ per step.
* `gamma` (`float = 0.9`): Future reward discount factor.
* `lam` (`float = 0.6`): Eligibility trace decay parameter $\lambda$.
* `eta` (`float = 0.15`): Learning rate for cortical weights $W$.
* `eta_v` (`float = 0.05`): Learning rate for affordance weights $V$ and $U$.
* `last_delta` (`float`): Most recent Temporal Difference error.

#### Methods
* `q(idx: np.ndarray, F: np.ndarray, gf: np.ndarray) -> np.ndarray`:
  $$Q(s, a) = \sum_{i \in \text{idx}} W[a, i] + \begin{cases} F[a] \cdot V & \text{if } a < 32 \\ U[a - 32] \cdot gf & \text{if } a \ge 32 \end{cases}$$
* `select(q: np.ndarray, eps: float, tau: float, rng: RandomState) -> int`:
  * Explores with probability $\epsilon$: $a \sim \text{Uniform}(0, N_{\text{actions}}-1)$.
  * Exploits via Boltzmann-Gibbs softmax: $P(a) \propto \exp\left(\frac{Q(s, a) - \max Q}{\tau}\right)$.
* `update(st, a, r, st_next)`:
  * TD Error: $\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$.
  * Trace Decay: $e \leftarrow \gamma \lambda e$.
  * Trace Potentiation: $e_W[a, \text{idx}] \mathrel{+}= 1.0$, $e_V \mathrel{+}= F[a]$ (or $e_U[a-32] \mathrel{+}= gf$).
  * Three-Factor Plasticity: $W \leftarrow \text{clip}(W + \frac{\eta}{k} \delta e_W, -4.0, 4.0)$, $V \leftarrow \text{clip}(V + \eta_v \delta e_V, -4.0, 4.0)$.

---

### 3.6 Class: `LanguageCortex`
Incremental Hebbian N-gram and semantic co-occurrence cortex with surprisal-gated learning.

#### Instance Variables & Attributes
* `w2i` (`Dict[str, int]`): Word token to integer ID vocabulary mapping (capped at 30,000 words).
* `i2w` (`Dict[int, str]`): Integer ID to word token inverted dictionary.
* `count` (`Dict[int, int]`): Total occurrences of each word.
* `bi` (`Dict[int, Dict[int, float]]`): Bigram associative weights $W(w_1 \to w_2)$.
* `bi_tot` (`Dict[int, float]`): Sum of outgoing bigram transitions from each source word.
* `tri` (`Dict[Tuple[int, int], Dict[int, float]]`): Trigram associative weights $W((w_1, w_2) \to w_3)$.
* `co` (`Dict[int, Dict[int, float]]`): Symmetric semantic co-occurrence graph ($+/- 4$ word window).
* `eta` (`float = 0.25`): Hebbian language plasticity rate.

#### Methods
* `surprisal(words: List[str]) -> float`:
  * Evaluates Laplace-smoothed bigram probability:
    $$P(w_t \mid w_{t-1}) = \frac{\text{bi}(w_{t-1}, w_t) + \epsilon / V}{\text{bi\_tot}(w_{t-1}) + \epsilon}$$
  * Average surprisal: $\bar{S} = -\frac{1}{N-1} \sum \log_2 P(w_t \mid w_{t-1})$.
* `ingest(sentence: str, strength: float = 1.0) -> int`:
  * Potentiates unigrams, bigrams, trigrams, and semantic co-occurrences scaled by `strength`.
* `_spread(topic: set) -> Dict[int, float]`:
  * Propagates spreading activation across the semantic co-occurrence network given topical seeds.
* `generate(seeds: List[str], n: int = 12, temp: float = 0.6) -> str`:
  * Synthesizes inner speech by blending trigrams (scale 4.0), bigrams (scale 1.0/2.5), spreading activation ($+30.0 \times \text{spread}$), and repetition penalties.
* `decay(f: float = 0.97, floor: float = 0.05)`:
  * Synaptic forgetting during sleep, pruning connections below `floor`.

---

### 3.7 Class: `Drives`
Homeostatic motivational drive engine.

#### Instance Variables & Attributes
* `boredom` (`float = 0.5`): Increases when sensory novelty is persistently low.
* `fatigue` (`float = 0.0`): Increases by $1 / \text{sleep\_every}$ per step; triggers sleep at $1.0$.
* `social` (`float = 0.0`): Jumps to $1.0$ on human communication; decays exponentially ($0.92$).
* `sleep_every` (`int = 400`): Duration of wakefulness phase in steps.

#### Methods & Properties
* `curiosity` (property): $C = \text{clip}(0.15 + 0.85 \times \text{boredom}, 0.15, 1.0)$.
* `update(novelty: float, human: bool)`: Adapts boredom, fatigue, and social motivation.
* `exploration(confidence: float) -> Tuple[float, float]`:
  * Returns $(\epsilon, \tau)$:
    $$\epsilon = 0.04 + 0.25 C (1.0 - 0.5 \times \text{confidence})$$
    $$\tau = 0.12 + 0.25 C$$
* `rest()`: Resets fatigue to 0 upon sleep completion.

---

### 3.8 Class: `GlobalWorkspace`
Global Workspace Theory (GWT) working memory stage.

#### Instance Variables & Attributes
* `items` (`List[Dict[str, Any]]`): Competing mental items `{"src": str, "words": List[str], "sal": float}`.
* `capacity` (`int = 4`): Maximum number of concurrent items allowed on stage.

#### Methods
* `post(src: str, words: List[str], sal: float)`: Adds a candidate representation.
* `tick()`: Decays salience by factor of $0.85$, prunes items below $0.05$, retains top `capacity`.
* `winner() -> Optional[Dict[str, Any]]`: Returns highest-salience item broadcast system-wide.

---

### 3.9 Class: `SelfModel`
Metacognitive monitoring tracking running stats and subjective confidence.

#### Instance Variables & Attributes
* `ema_r` (`float = 0.0`): Exponential moving average of reinforcement reward ($\alpha = 0.02$).
* `ema_nov` (`float = 0.5`): Exponential moving average of novelty ($\alpha = 0.02$).
* `ema_err` (`float = 0.15`): Exponential moving average of forward prediction error ($\alpha = 0.02$).
* `counts` (`np.ndarray`, shape: `(37,)`): Action execution frequency distribution.

#### Methods & Properties
* `observe(a: int, r: float, nov: float, err: float)`: Updates moving statistics.
* `confidence` (property): $\text{Conf} = \frac{1}{1 + 10 \times \text{ema\_err}} \in (0, 1]$.
* `report() -> Dict[str, float]`: Telemetry summary including Shannon action entropy:
  $$H(A) = -\frac{\sum p_i \ln p_i}{\ln 37}$$

---

### 3.10 Class: `Organism`
Unified cognitive agent integrating perception, memory, policy, language, and action into a closed loop.

#### Complete Step-by-Step Cognitive Loop (`step()`)
1. **Sensory Ingestion**: Obtains RGB frame and motion flux from environment.
2. **Retinal Processing**: Computes 6-channel maps, 870D holistic scene $x$, and $32 \times 8$ local affordances $F$.
3. **Cortical Encoding**: $x \to \text{idx}$ (24 sparse active units).
4. **Episodic Recall**: Queries prototype memory $\to (\text{ctx}, \text{nov}, \text{sim})$.
5. **Epistemic Update & Reward Formulation**:
   * Evaluates forward model prediction error: $\text{err} = |g_t - \hat{g}_t|$.
   * Computes perceptual change: $\text{change} = \text{clip}\left(\frac{1 - x_t \cdot x_{t-1} - 0.05}{0.5}, 0, 1\right)$.
   * Detects wasted motor effort: $\text{wasted} = (x_t \cdot x_{t-1} > 0.985)$.
   * Blends composite multi-factor reward:
     $$r = 0.55 \times \text{nov} + 0.15 \times \text{change} \times \text{habit} + 0.40 \times r_{\text{text}} + 0.25 \times \text{progress} - 0.08 \times \mathbf{1}_{\text{wasted}} + r_{\text{human}}$$
6. **Policy Plasticity**: Updates $W$, $V$, $U$ via TD(lambda) Three-Factor Hebbian rule.
7. **Inhibition of Return**: Ineffective clicks on a scene are added to `dead[ctx]`, suppressing repeat attempts.
8. **Action Selection**: Evaluates $Q(s, a) + 0.5 Q_e[\text{ctx}, a]$, applies dead action penalties, and selects action via Boltzmann-Gibbs softmax.
9. **Motor Execution**: Invokes environment action (`click`, `scroll`, `read`, `type`, `back`).
10. **Conscious Broadcasting & Inner Speech**: Workspace winner seeds associative thought generation via `lang.generate()`.
11. **Sleep Check**: If fatigue $\ge 1.0$, triggers `sleep()`.

---

## 4. Synthetic Test Environment: `MockWeb`

`MockWeb` provides a procedural benchmark environment for evaluating visual navigation, reading, and learning dynamics:
* **Procedural Textures**: Kronecker-scaled $9 \times 16 \times 3$ RGB arrays forming chunky, blocky visuals.
* **Vertical Layout Stripes**: Simulates page text/column layout.
* **Clickable Links**: 3 high-contrast blue rectangles placed at specific fixation cells.
* **Hyperlink Graph**: Clicking a link transitions to target pages with 95% physical reliability.
* **Natural Corpus**: 16 scientific and historical sentences ingested upon `A_READ` actions.

---

## 5. Verification Suite & Empirical Benchmarks

The self-test suite (`python3 core.py --selftest`) verifies:
1. **Link Generalization**: The trained TD-Hebbian learner achieves $>2\times$ higher effective click rates compared to random exploration baselines.
2. **Web Coverage Efficiency**: The learner discovers 20 distinct pages in $<60\%$ of the steps required by random walk.
3. **Language Acquisition**: Automatically expands vocabulary from zero to dozens of words from sensory text.
4. **Surprisal Habituation**: The agent finds repeatedly seen sentences significantly less surprising ($<2.0$ bits) than novel out-of-distribution text ($>10.0$ bits).
5. **Inner Monologue**: Generates syntactically and semantically coherent internal monologue from learned associations.
6. **State Persistence**: 100% round-trip lossless serialization and deserialization of all synaptic matrices and prototype memories via compressed NPZ checkpoints.
7. **Sleep Consolidation**: Successful replay of episodic transitions, synaptic pruning of weak language associations, and agglomerative merging of duplicate prototypes.
