# Sub-feature decisions

Locked design decisions for each sub-feature of ASL-live, captured during
sub-feature-by-sub-feature design review. Companion document to:
- [`discussion.md`](discussion.md) — high-level architecture
- [`plans/PLAN.md`](plans/PLAN.md) — phase-by-phase implementation plan

A sub-feature is "locked" once both parties have signed off on every open
question for it. Locked decisions are not re-litigated unless new evidence
appears (e.g., a confusion matrix shows the choice was wrong).

---

## #1 — Hand-landmark extraction ✅ LOCKED

**Module:** `src/asl_live/recognition/landmarks.py`

**Public API:**
```python
def extract(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns a normalized 63-dim landmark vector for the most confident hand,
    or None if no hand is detected.
    The returned vector is always in canonical "right-hand" form (left-hand
    detections are mirrored).
    """
```

### Decisions

1. **Multi-hand handling.** If MediaPipe detects more than one hand, pick
   the single most-confident detection and ignore the others. No error,
   no warning — silent.

2. **Handedness normalization.** MediaPipe labels each detection as left
   or right. Left-hand detections are mirrored (x → -x in image space)
   *before* normalization, so the classifier only ever sees a canonical
   right-handed sign. Rationale: avoids doubling the dataset / risk of
   overfitting on a left-vs-right split.

3. **No-hand return value.** `extract()` returns `None` when no hand is
   detected. Downstream code (debounce, recognizer worker) treats `None`
   as a "gap" frame. **The classifier is not invoked on `None` frames** —
   this is the efficiency contract.

4. **Normalization scheme.** Two steps, in order:
   - Translate: subtract wrist (landmark 0) from every landmark → wrist at origin.
   - Scale: divide all coordinates by the maximum wrist-to-landmark distance
     in the frame → the largest finger reach becomes magnitude 1.

   **No rotation normalization** and **no distance-only features**.
   Rationale: some ASL letter pairs are distinguished by hand rotation /
   tilt; collapsing rotation away would conflate them.

5. **MediaPipe `model_complexity`.** Default `1` (more accurate). Fall back
   to `0` (faster) only if the Pi can't sustain ≥ 15 fps in phase 3
   testing.

6. **Input resolution.** Camera captures at 640 × 480. MediaPipe internally
   resizes; sending higher res is wasted compute.

### Out of scope for this sub-feature

- Multi-frame temporal features (would be needed for J/Z motion signs —
  explicitly out of scope per discussion §1).
- Rotation-invariant feature extraction (rejected above).
- 3D pose estimation beyond MediaPipe's built-in z coordinate.

---

## #2 — Data collection ✅ LOCKED

**Modules:**
- `scripts/ingest_public.py` — one-shot preprocessor for public dataset.
- `src/asl_live/capture/collect.py` — interactive collection of custom gestures.

### Decisions

1. **Primary dataset: Kaggle ASL Alphabet** (grassknoted). ~87,000 labeled
   images, 29 classes, 200×200 resolution, permissive license (GPL-2.0).
   - Use classes A–Y for letters.
   - Discard J and Z (out of scope per discussion §1).
   - Discard the dataset's "space", "delete", "nothing" classes — their hand
     shapes don't match our chosen gestures.

2. **Two-script collection pipeline.**
   - `ingest_public.py` runs MediaPipe over each Kaggle image, extracts the
     63-dim landmark vector via the same `landmarks.extract()` used at
     runtime, drops frames where no hand was detected, saves as `.npy`
     under `data/landmarks/<class>/`. One-shot, ~1–2 h of CPU.
   - `collect.py` is interactive, only used for SPACE, DELETE, and
     optional user-specialization top-ups.

3. **Custom-collection cadence.** Auto-capture while the hand is detected
   and the landmarks are stable for 5 frames, with a 10-frame cooldown
   between captures. Manual key-press capture not used.

4. **Diversity prompts during custom collection.** Skipped for SPACE and
   DELETE — the gestures are gross enough that pose variation matters
   little. Re-introduce only if a future top-up is needed for fine letter
   distinctions.

5. **Where to collect.** PC, not Pi. Faster iteration. Pi top-up only if
   phase 2 evaluation shows poor real-world accuracy.

6. **Mirror augmentation.** Applies to *both* public ingest and custom
   collection. Each saved sample is also stored mirrored (x → -x in
   normalized coordinates), effectively 2× free data and consistent with
   the runtime extractor's left → right mirroring (sub-feature #1).

7. **Storage format.** Per-sample `.npy` files under
   `data/landmarks/<class>/<source>_<id>.npy` (so we can tell public from
   custom and delete bad samples easily). Consolidated to a single
   `data/dataset.npz` at the start of every training run.

8. **Raw frames.** Not saved by default. `collect.py` accepts a
   `--save-frames` flag for occasional debugging. Public ingest never
   saves frames — landmarks only.

9. **Target class sizes.**
   - Letters A–Y: ~3,000 samples each from public ingest.
   - SPACE, DELETE: ~500 samples each from custom collection.
   - Class imbalance handled at training time via class-weighted
     cross-entropy (sub-feature #3).

10. **License posture (v1).** GPL-2.0 of the source dataset is acceptable
    for the prototype. Documented as a dependency in README. Re-evaluate
    if the project ever moves toward redistribution.

11. **Gitignore.** `data/`, `models/`, `__pycache__/`, `*.pyc`, `.venv/`,
    `.env` — added when implementation starts.

### Out of scope for this sub-feature

- Multi-hand or bimanual sign data.
- Video sequences (would be needed for J/Z motion signs).
- Synthetic data generation.

---

## #3 — Classifier ✅ LOCKED

**Modules:**
- `src/asl_live/train/train_mlp.py` — training script (PC).
- `src/asl_live/recognition/classifier.py` — runtime inference wrapper (Pi).
- `models/mlp.onnx` — exported model.
- `models/label_map.json` — index → class name lookup.
- `models/training_report.json` — full record of the training run.

**Public API (runtime):**
```python
def predict(landmarks: np.ndarray) -> tuple[str, float]:
    """Return (class_name, confidence). Only invoked when landmarks is not None."""
```

This sub-feature is documented in more depth than #1 and #2 because the
user is learning the ML stack alongside the project — the doc serves as
both decision record and study reference.

### 3.1 — Architecture: 4-layer MLP, with XGBoost as a baseline

**Problem.** We need a function that maps a 63-dim normalized landmark
vector to one of 26 classes, runs in microseconds on the Pi, trains in
minutes on a PC, and is small enough to ship.

**Solution.** A 4-layer Multi-Layer Perceptron (MLP):

```
[63 inputs] → [128, ReLU, Dropout 0.2] → [64, ReLU] → [26 outputs] → softmax
```

**How it works.** Each "layer" is a fully connected (dense) layer: every
input is multiplied by a learned weight, the products are summed at every
neuron, and a non-linearity is applied. Stacking layers lets the network
learn complex patterns.

- **ReLU** (`max(0, x)`) is the activation function. It zeroes out
  negative values. This tiny non-linearity is what makes deep networks
  meaningful — without it, stacking layers would collapse into a single
  linear layer mathematically.
- **Dropout 0.2** randomly turns off 20 % of neurons each training step.
  This forces the network to spread information across many neurons rather
  than over-rely on any single one. Standard regularization to prevent
  overfitting.
- **Softmax** at the end converts the 26 raw output numbers ("logits")
  into probabilities that sum to 1.0. The argmax is the predicted class;
  the value at that index is our confidence.

Total parameters: ~30,000. Total disk size at FP32: ~120 KB. Inference per
frame: < 5 µs.

**Why these specific layer sizes (128, 64).** Enough capacity for a
63-input, 26-output problem on ~80 k training examples, but small enough
that the model is unlikely to memorize the dataset. Bigger layers risk
overfitting and waste Pi compute; smaller layers risk underfitting.

**Why an MLP and not a CNN, RNN, or Transformer.**
- **CNN** (Convolutional Neural Network) is for images — detecting local
  spatial patterns. We're not feeding images, we're feeding 63 abstract
  numbers. A CNN buys us nothing.
- **RNN / LSTM / GRU** is for sequences over time. We classify single
  frames; there's no temporal sequence at this stage. (Sub-feature #4
  handles temporal stability.)
- **Transformer** is for sequences with attention. Same reason as RNN.
  Massive overkill for tabular classification.
- An **MLP** is the de facto choice for low-dim tabular classification.
  Right tool, right size.

**XGBoost as a baseline (not deployed).**

XGBoost is gradient-boosted decision trees: a sequence of small decision
trees where each new tree is trained to fix the previous trees' errors,
and the final prediction is the sum of all tree outputs.

On *tabular* data XGBoost is famously strong — it often beats neural
networks of comparable size, with no GPU and no hyperparameter tuning.
It's the standard baseline for tabular classification.

We train it once, with default hyperparameters, alongside the MLP. The
comparison gives us:
- **MLP wins or ties XGBoost**: ship MLP with confidence; the
  architecture is appropriate.
- **XGBoost beats MLP by > 1 % macro-F1**: the MLP is misdesigned (likely
  needs more capacity, different features, or different regularization);
  diagnose and re-train MLP before shipping.

We do **not** deploy XGBoost. It exists only as a sanity check measuring
stick. Cost: ~30 seconds of training time.

### 3.2 — Loss function: class-weighted cross-entropy

**Problem.** Our dataset is imbalanced — ~3,000 samples per letter from
the public ingest, but only ~500 each for SPACE and DELETE from custom
collection. With a uniformly-weighted loss, the optimizer learns to "get
the common classes very right and the rare ones somewhat wrong" because
that minimizes the average loss. Net effect in real use: the device fails
specifically on the control gestures the user signs constantly, ruining
the UX.

**Background: cross-entropy.**

A loss function is the algorithm's measurement of "how wrong" the model
is on a training example, so it can adjust weights to be less wrong. For
classification, the standard choice is **cross-entropy**:
- The model produces a probability for each class via softmax.
- Cross-entropy looks at the probability the model gave to the *correct*
  class.
- High probability for the correct class → small loss.
- Low probability for the correct class → large loss.

The optimizer averages cross-entropy across the batch and steps weights
to reduce that average.

**The imbalance failure.**

```
Total loss ≈ Σ over all samples (per-sample cross-entropy)
           = (3000 × loss on As) + (3000 × loss on Bs) + ...
           + (500 × loss on SPACEs) + (500 × loss on DELETEs)
```

The ~3,000-sample classes dominate the sum. Even sloppy SPACE accuracy
barely moves the average loss, so the optimizer never bothers to fix it.

**Solution: per-class weights, inversely proportional to class frequency.**

```python
weights = total_samples / (num_classes * samples_per_class)
loss = nn.CrossEntropyLoss(weight=weights)
```

A SPACE sample's loss × ~6.0; a letter sample's loss × ~1.0. Now SPACE
mistakes cost the optimizer 6× more than letter mistakes. The gradient
signal is balanced even though the dataset isn't.

**Why not the alternatives.**
- **Oversampling** (duplicate minority samples until counts match) works,
  but the model sees the same SPACE sample 6× per epoch and can overfit
  to that exact 500-sample collection.
- **Focal loss** down-weights easy examples and up-weights hard ones —
  designed for *extreme* imbalance (1:100 or worse, e.g., object
  detection's background pixels). Our 6:1 imbalance is mild; focal loss
  is unnecessary complexity.
- **Class weighting** is one parameter on `nn.CrossEntropyLoss`,
  well-understood, no overfitting risk. Standard textbook fix.

### 3.3 — Augmentation: noise + scale + translation

**Problem.** Without augmentation, the model can memorize specific
landmark vectors instead of learning the underlying gesture shape. In
real use the user's hand is slightly closer / farther from the camera,
slightly off-centered in the frame, and MediaPipe's detector returns
slightly different landmarks even on a still hand. The unaugmented model
misclassifies these natural variations.

**Solution.** At training time, each sample has an independent 50 %
chance of receiving each of three perturbations (so most samples get one
or two; some get all three; some get none).

| Perturbation | Magnitude | Simulates |
|---|---|---|
| Gaussian noise on every coordinate | std = 0.01 | MediaPipe detection jitter frame-to-frame |
| Uniform scale | × random in [0.95, 1.05] | User closer / farther from camera |
| Translation on x, y | ± random in [-0.02, 0.02] | Hand framed differently in the camera view |

(Coordinates are normalized so wrist-relative magnitudes sit roughly in
[-1, +1]; the augmentation magnitudes are calibrated to that range.)

```python
def augment(landmarks):
    if random() < 0.5:
        landmarks += gaussian_noise(std=0.01, shape=landmarks.shape)
    if random() < 0.5:
        landmarks *= uniform(0.95, 1.05)
    if random() < 0.5:
        landmarks[:, :2] += uniform(-0.02, 0.02, size=2)
    return landmarks
```

**Augmentation is randomized per epoch**, so every pass through the
dataset shows the model slightly different versions of each sample. This
effectively expands the training set for free. Augmentation is **not**
applied to val/test data — those need to reflect "real" performance.

**What we explicitly do NOT augment.**
- **No rotation augmentation.** Sub-feature #1 decided that rotation
  matters for distinguishing some sign pairs; augmenting it would teach
  the model to ignore rotation, breaking those distinctions.
- **No flip / mirror augmentation.** Mirror flips happen at data ingest
  time (sub-feature #2). Doing it again at training time would
  double-count, doubling the model's belief that left and right hands
  are interchangeable (which we already established).

**Why this matters more for SPACE/DELETE than for letters.** The public
dataset has ~3,000 letter samples each from many contributors with
different hands, lighting, and camera angles — natural variation built
in. Our 500 self-collected SPACE/DELETE samples are way more uniform;
augmentation is the mechanism that gives them the variety needed to
generalize beyond your specific captures.

### 3.4 — Train / val / test split: 80 / 10 / 10 stratified, seed 42

**Problem.** If we measure the model on the same data we trained it on,
the result is meaningless — the model has effectively seen the answers.
Worse, if we tune hyperparameters by repeatedly looking at the test
set, we'd indirectly leak the test set into our decisions, and our
final reported number wouldn't honestly predict real-world behavior.

**Solution.** Three disjoint splits with three distinct roles.

| Split | Size | Role | When touched |
|---|---|---|---|
| Train | 80 % | Optimizer adjusts weights against this | Every training step |
| Val | 10 % | Early stopping, LR schedule, hyperparameter peeking | After each epoch, freely |
| Test | 10 % | Final reported number for the run | **Exactly once**, at the very end |

**Why three and not two.** Val is the layer between the model and the
test set. We can iterate against val all we want — it's the "sandbox."
The test set stays untouched, so its number is an honest estimate of
how the model will perform on data it has truly never seen. This
convention is universal in ML papers and code.

**"Stratified" means the split preserves class proportions.** A naive
random split could (by bad luck) put almost all SPACE samples into
train, leaving val with 30 SPACE samples — metrics on that tiny group
are too noisy to trust. Stratification guarantees each split contains
the same class proportions as the full dataset. One parameter on
`sklearn.model_selection.train_test_split`.

**Random seed 42** so the same split is produced every run. Without a
fixed seed, comparing two training runs becomes meaningless — we
wouldn't know whether the difference was the change we made or just
luck of the split.

**Known leakage risk (accepted for v1).** The Kaggle ASL Alphabet has
near-duplicate frames from the same recording session; some near-dupes
may end up split across train and val/test, making val/test scores
optimistic. Splitting *by source* would fix this but the dataset
doesn't expose source metadata cleanly. We accept the risk for v1 — the
*real* generalization test is the phase-3 live demo on the user's own
camera, not test-set numbers. The training report flags this caveat.

### 3.5 — Hyperparameters

| Knob | Value | Why |
|---|---|---|
| Optimizer | **Adam** | Adaptive, works well out-of-the-box; see below |
| Learning rate | 1e-3 (0.001) | Adam's safe default for almost any project |
| Weight decay | 1e-4 | Light regularization, pushes weights toward zero |
| Batch size | 256 | Smooth gradients, fits in memory comfortably |
| Max epochs | 100 | Generous budget; early stopping ends earlier |
| Early stopping patience | 10 epochs | Halt if val loss hasn't improved for 10 epochs |
| LR schedule | ReduceLROnPlateau, factor 0.5, patience 5 | Halve LR when val loss plateaus |
| Random seed | 42 | Reproducibility |

**About Adam (Adaptive Moment Estimation).**

The optimizer is the algorithm that updates model weights given the
current loss. The simplest is **SGD** (stochastic gradient descent):
take the gradient of the loss with respect to each weight, step every
weight by `learning_rate × gradient`. Works, but is slow, finicky, and
sensitive to learning-rate choice.

Adam improves on SGD with two tricks:

1. **Per-weight adaptive learning rates.** Adam tracks how much each
   weight has been changing recently. Weights that change a lot get a
   smaller effective step (so they don't overshoot); weights that
   rarely move get a larger one (so they're not stuck). The single
   global `lr` is automatically adapted per individual weight.
2. **Momentum.** Adam keeps a running average of recent gradients (the
   "first moment") and recent squared gradients (the "second moment").
   These smooth out the direction of motion through noisy gradient
   estimates. Picture a heavy ball rolling downhill instead of a feather
   buffeted by wind — the ball ignores small bumps and follows the
   underlying slope.

Net effect: Adam reaches a good model faster than SGD and is much less
sensitive to the specific learning-rate value. For ~95 % of small/medium
projects, Adam at `lr = 1e-3` works on the first try.

**About early stopping.** Validation loss almost always decreases
initially, flattens, then starts *increasing* as the model overfits the
training data. Early stopping detects the start of that climb and halts
training, keeping the best checkpoint. Patience = 10 means "tolerate 10
consecutive epochs without improvement before halting" — a buffer
against noisy plateaus.

**About `ReduceLROnPlateau`.** When val loss stops improving, halving
the learning rate sometimes lets the optimizer find finer minima it was
overshooting. Cheap to add and rarely hurts. Patience = 5 means we
halve LR after 5 stagnant epochs (faster than early stopping's 10).

**No automated hyperparameter sweep for v1.** We run once with these
defaults, look at the confusion matrix, and only tune if results are
poor. Sweeping (e.g., with Optuna over LR, hidden sizes, dropout) costs
10–50× the compute and rarely changes the outcome on small models like
ours. Manual iteration after the first run, if needed.

### 3.6 — Acceptance metrics: macro-F1 and confusion matrix

**Problem.** Plain accuracy ("% correct") is misleading on imbalanced
classes. With 600 letters and 100 SPACE/DELETE samples in the test set,
a model that *always predicts a letter* and ignores SPACE/DELETE
entirely scores ~85 % accuracy — and is useless. We need metrics that
don't let minority-class failures hide behind majority-class wins.

**Per-class precision and recall — two distinct concerns.**

For each class (e.g., SPACE), we ask two different questions:

- **Precision**: of all the times the model *said* "SPACE", how many
  were *actually* SPACE?
- **Recall**: of all the actual SPACE samples, how many did the model
  *catch*?

These have different real-world consequences in our device:

| Failure mode | What goes wrong | Concrete example |
|---|---|---|
| **Low precision on SPACE** | Device fires SPACE when you didn't sign it | You're mid-signing "HELLO"; mid-letter the model briefly thinks it sees an open palm; word commits as "HEL" |
| **Low recall on SPACE** | Device misses SPACE you actually signed | You sign SPACE; device shows "HELLO" but never sends; you keep signing into the void |

Both matter. Both are tracked per class.

**F1 — harmonic mean of precision and recall.**

```
F1 = 2 × (precision × recall) / (precision + recall)
```

The harmonic-mean structure punishes lopsided performance: precision 1.0
+ recall 0.1 gives F1 ≈ 0.18, not the (1.0+0.1)/2 = 0.55 you'd get from a
simple average. F1 is high *only when both* precision and recall are high.

**Macro-F1 — unweighted average of per-class F1.**

We compute F1 for each of the 26 classes individually, then average
without weighting. The "macro" qualifier matters: an alternative
("micro-F1") weights samples equally and brings back the imbalance
problem (3,000 letter samples drown out 500 SPACE samples). Macro-F1
forces the model to be good at *every* class regardless of frequency —
exactly what our use case needs.

**Confusion matrix.**

A 26 × 26 table: rows = actual class, columns = predicted class. Each
cell counts how often that combination occurred. Diagonal cells =
correct predictions; off-diagonal cells = mistakes broken down by
exactly which two classes were confused.

```
              Predicted →
              A    B    ...   SPACE  DELETE
Actual A     580   3    ...    1      2
Actual B      2  595    ...    0      1
...
Actual SPACE  3   0     ...   485     2
Actual DEL    7   0     ...    0    488
```

Macro-F1 says *how good* the model is overall. The confusion matrix
says *exactly where it fails*. If "Actual DELETE → Predicted A" shows 7
out of 50 (14 %), that's the early signal that thumb-down DELETE looks
too much like the letter A; we'd switch DELETE to a pinch gesture
*before* phase 3.

**The acceptance bar (both must hold to advance from phase 2 to 3):**

1. **Macro-F1 ≥ 0.95** on the test set.
2. **No off-diagonal cell exceeds 2 % of its row total** — for any
   actual class, no specific wrong prediction happens more than 2 % of
   the time.

Either fail → don't advance. Iterate by collecting more data for the
weak class, swapping a confusing gesture, or adjusting the model.

### 3.7 — Confidence threshold (`MIN_CONF`)

**Problem.** The MLP always returns *some* prediction, even on
ambiguous frames (mid-transition between letters, weird angles, the
hand half-closed). On those frames the top probability might be 0.4 —
barely better than random. If we let those low-confidence guesses
commit to the word buffer, random characters appear during natural
gesture transitions.

**Solution.** The runtime debounce (sub-feature #4) only counts a frame
as "stable" if its top-class probability is ≥ `MIN_CONF`. Below the
threshold, the frame is treated as uncertain and ignored.

**Default: `MIN_CONF = 0.85`.** Picked from common practice. Tuning
trade-off:
- Higher (0.95) → fewer false commits but slower (need cleaner signing).
- Lower (0.7) → faster commits but more spurious characters.

Re-tuned in phase 7 polish based on observed false-commit and
missed-commit rates from real use.

**Why we don't formally calibrate.** Softmax probabilities are usually
**overconfident** — when a model says "0.95", real-world correct rate
at that confidence is more like 0.85. Formal fixes (e.g., temperature
scaling on val data) exist but are overkill for v1; empirical tuning is
good enough.

### 3.8 — ONNX export: opset 17, dynamic batch axis

**Problem.** We train the MLP in PyTorch on the PC. We can't ship
PyTorch to the Pi — it's hundreds of MB, slow to import, heavy memory
footprint — and we don't want to.

**Solution.** Export the trained MLP to a portable file format that the
Pi can load with a tiny runtime.

**Key clarification: ONNX is a file format, not a model.** It's a
container, like JPEG for images. The same trained MLP could be saved as
a PyTorch `.pt` file or a TFLite `.tflite` file or an ONNX `.onnx`
file — same content, different containers. The MLP doesn't change; only
how it's packaged for transport changes.

**Why ONNX over TFLite.**
- The runtime (`onnxruntime`) is a single ~50 MB Python package.
- Works on Pi 5 with no extra dependencies.
- Better Python ecosystem (numpy interop, debugging tools).
- TFLite is better suited to mobile / accelerator hardware (Coral,
  NPUs) — irrelevant for plain Pi 5 CPU.

**Opset 17.** ONNX evolves and adds operators over time; each version
is called an *opset*. Picking opset 17 explicitly guarantees the file
we export today loads in any modern `onnxruntime`. Mainstream and
stable. No downside.

**Dynamic batch axis.** The exported model accepts any batch size at
inference. We always run batch 1 on the Pi (one frame at a time), but
flexibility is free.

**Export call (one line at the end of training):**
```python
torch.onnx.export(
    model, dummy_input, "models/mlp.onnx",
    opset_version=17,
    input_names=["input"],  output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
)
```

**Inference at runtime (Pi side):**
```python
import onnxruntime as ort
session = ort.InferenceSession("models/mlp.onnx")
output = session.run(None, {"input": landmark_vector})
predicted_class = np.argmax(output[0])
```

That's the entire deployment surface. PyTorch is never installed on the
Pi.

### 3.9 — Quantization: skipped for v1

**The problem (which we don't actually have).** Large models are slow
and bloated on constrained hardware.

**The quantization technique.** Compress weights from 32-bit floats
(FP32) to 8-bit integers (INT8). Cuts model size 4× and often speeds
inference 2–3× on CPUs. Trades a small accuracy hit (typically < 1 %)
for much smaller, faster models.

Two flavors:
- **Post-training quantization (PTQ)** — quantize a trained FP32 model
  in one extra step. Easy, but accuracy hit can be larger.
- **Quantization-aware training (QAT)** — simulate the rounding *during*
  training so the model learns to be robust. Better accuracy, much more
  code.

**Why we skip it.** Our MLP is 30 k parameters / 120 KB at FP32. INT8
would save 90 KB and a few microseconds per inference. The Pi 5 has
8 GB RAM and a fast quad-core CPU — orders of magnitude more headroom
than we need. Adding quantization code would be pure overhead with no
user-visible benefit.

**When we'd revisit.** If phase-3 profiling shows the classifier is the
bottleneck (it won't be — MediaPipe will dominate). Profile first,
optimize second.

### 3.10 — Training artifacts

Every training run writes three files to `models/`:

1. **`mlp.onnx`** — the deployable model.
2. **`label_map.json`** — `{0: "A", 1: "B", ..., 24: "SPACE", 25: "DELETE"}`.
   Without this, the ONNX file's outputs are nameless integers.
3. **`training_report.json`** — git commit hash, timestamp,
   hyperparameters, dataset stats, macro-F1, per-class F1, full
   confusion matrix as a nested dict, MLP-vs-XGBoost comparison.

**Why all three, every run.** When iterating ("did adding more SPACE
samples help? did the new augmentation regimen help?"), we constantly
compare runs. Without the report, models are opaque blobs we can't
reconstruct or interpret. The git commit hash means we can always check
out the exact source code that produced any model.

### Out of scope for this sub-feature

- Ensemble methods (combining multiple models at inference).
- Alternative export formats (TFLite, plain PyTorch).
- Quantization (revisit only if profiling demands it).
- Automated hyperparameter sweeps (manual iteration only in v1).
- Sequence models (would be needed for J/Z motion signs — out of scope
  per discussion §1).
- Confidence calibration via temperature scaling (overkill for v1).

---

## #4 — Debounce / commit logic ✅ LOCKED

**Module:** `src/asl_live/recognition/debounce.py`

**Public API:**
```python
@dataclass
class CommitEvent:
    kind: Literal["LETTER", "SPACE", "DELETE"]
    letter: Optional[str]   # only set when kind == "LETTER"
    confidence: float

class Debouncer:
    def step(self, prediction: Optional[tuple[str, float]]) -> Optional[CommitEvent]:
        """
        Feed one frame's classifier output. Returns a CommitEvent on the
        frame that triggers a commit, otherwise None. Pass `None` for
        frames where the landmark extractor returned no hand.
        """
```

### 4.1 — Problem

The classifier produces a `(class, confidence)` tuple per frame at
15–30 fps. A single held sign produces dozens of nearly-identical
predictions. Frames during gesture transitions (when the hand is
mid-shape) produce noisy, low-confidence, or simply wrong predictions.

We need to convert this messy stream of per-frame predictions into
discrete *commit events* — exactly one event per intentional sign —
without:
- committing 30 copies of "A" while the user is just holding A,
- committing spurious letters from transition frames,
- missing the user's intended next letter.

### 4.2 — Design: single counter + blind cooldown

The locked design is the simplest one that does the job. Three pieces of
internal state:
- `current_class` — the class of the running streak (None if broken).
- `streak` — consecutive frames matching `current_class`.
- `cooldown` — frames remaining in a blind cooldown after the last
  commit.

Per-frame logic:

1. **If cooldown > 0**: decrement and return None. We ignore every input
   during cooldown — it doesn't matter whether the classifier sees the
   same class, a different class, or nothing.
2. **Otherwise, if input is None or below `MIN_CONF`**: reset streak,
   return None.
3. **Otherwise**: if class matches the running streak, increment streak;
   if not, restart streak at 1 with the new class.
4. **If streak reaches `STABLE_FRAMES`**: commit the class, set
   `cooldown = GAP_FRAMES`, reset streak, return the `CommitEvent`.

That's the entire algorithm. ~25 lines including the dataclass.

### 4.3 — Why this is sufficient (and the rejected alternatives)

**Rejected: explicit two-state machine (WATCHING / COOLDOWN).** Earlier
draft. Adds nothing — the "states" are just whether `cooldown == 0` or
not. Removed.

**Rejected: track a per-frame "gap" condition (no-hand or different
class).** Earlier draft. Required defining "what counts as a gap frame"
and led to subtle edge cases when the user holds the same letter and
then signs it again. The blind cooldown is simpler and behaves the same
in normal use.

**Rejected: majority-within-sliding-window for stability detection.**
Tolerates classifier flicker (e.g., A,A,A,B,A,A,A) more gracefully than
strict consecutive matching, but adds a window length parameter and
hides bugs in the classifier behind tolerant logic. We use **strict
consecutive** for v1: any non-matching frame resets the streak. If
phase-3 testing reveals real flicker problems we'll switch — but most
likely the classifier is good enough that strict works.

### 4.4 — Trade-off introduced by the blind cooldown

Because the cooldown is blind, after a commit the user must **move the
hand within `STABLE_FRAMES + GAP_FRAMES` frames** or the same letter
will commit a second time. With defaults (5 + 3 = 8 frames at 30 fps =
~270 ms) this is tight but reasonable for ASL fingerspelling, which
naturally transitions between shapes.

This is also what makes **repeated letters** (the two L's in "HELLO")
work cleanly without requiring the user to lift their hand — they just
need to make any small transition during the cooldown. With the
previous "gap requires no-hand or different class" design, repeated
letters required a hand-lift, which is unnatural for ASL.

The realistic tuning is probably `GAP_FRAMES = 10–15` (~330–500 ms) once
we test live. Default stays at 3 until phase 7.

### 4.5 — Parameters

| Parameter | Default | Source / role |
|---|---|---|
| `STABLE_FRAMES` | 5 | How many consecutive same-class frames before commit. ~167 ms at 30 fps. |
| `GAP_FRAMES` | 3 | Blind cooldown after commit. ~100 ms at 30 fps. |
| `MIN_CONF` | 0.85 | Locked from sub-feature #3.7. Frames below this are treated as "no signal." |

All three live in `config.py` and are tuned empirically in phase 7. The
debounce module reads them at construction time.

### 4.6 — Behavior trace: signing "HELLO" (STABLE=5, GAP=3)

| Frames | Input | State after | Result |
|---|---|---|---|
| 1–5 | H, H, H, H, H | streak=5 → commit | **emits LETTER('H')**, cooldown=3 |
| 6–8 | (anything) | cooldown ticks down | None |
| 9–13 | E, E, E, E, E | streak=5 → commit | **emits LETTER('E')**, cooldown=3 |
| 14–16 | (transition / hand still moving) | cooldown ticks down | None |
| 17–21 | L, L, L, L, L | streak=5 → commit | **emits LETTER('L')**, cooldown=3 |
| 22–24 | L, L, L (user still holding L) | cooldown ticks down | None |
| 25–29 | L, L, L, L, L | streak=5 → commit | **emits LETTER('L')** (second L), cooldown=3 |
| 30+ | … O eventually | … | **emits LETTER('O')** when streak hits 5 |

Note that the second L commits *without* the user lifting their hand —
they just need to keep holding L past the cooldown.

### 4.7 — Edge-case behavior (locked)

**Empty buffer + SPACE / DELETE.** The debounce just emits the event;
the *recognizer worker* is responsible for handling these:
- SPACE on empty buffer → silent no-op (no translation triggered).
- DELETE on empty buffer → silent no-op.
- Rationale: don't punish accidental signs with audible feedback.

**Held ambiguous sign with no clear class.** Confidence stays below
`MIN_CONF`, streak never grows, no commit ever fires. The Debouncer
simply waits. No timeout, no give-up logic — the user makes a clearer
sign and it commits.

**Classifier flicker (A,A,A,B,A,A,A).** Strict consecutive matching: the
single B resets the streak to 1. The next A starts a fresh streak. If
the classifier is well-trained (per #3 acceptance bar) this is rare.

### 4.8 — Testability

`Debouncer.step()` is a pure function over the prediction stream — no
I/O, no globals, no clock. It can be unit-tested by feeding canned
streams and asserting the emitted events. Tests live in
`tests/test_debounce.py` and run in milliseconds without camera or
model.

Required test cases:
1. 5 × ("A", 0.95) → emits LETTER('A') exactly once.
2. 5 × ("A", 0.95) followed by 8 × None → still exactly one
   LETTER('A') (cooldown then idle).
3. 5 × ("A", 0.95), 3 × None, 5 × ("A", 0.95) → emits LETTER('A')
   twice (cooldown elapsed, fresh streak).
4. 4 × ("A", 0.95), 1 × ("B", 0.95), 5 × ("A", 0.95) → emits
   LETTER('A') exactly once (the B reset the streak).
5. 5 × ("A", 0.5) → no emit (below MIN_CONF).
6. 5 × ("A", 0.95) followed by 30 × ("A", 0.95) without any
   transition → emits LETTER('A') ~3 times (cooldown elapses,
   streak rebuilds, recommits — documented behavior).

The tests are written *before* the implementation, per phase 3 of
PLAN.md.

### Out of scope for this sub-feature

- Multi-frame motion features (would be needed for J/Z signs).
- Adaptive thresholds based on signing speed.
- Confidence-weighted commits (e.g., averaging confidence across the
  streak).
- "Soft commit" UI that previews the imminent commit before it happens.

---

## #5 — Translation (Argos) ✅ LOCKED

**Module:** `src/asl_live/translation/translator.py`
**Setup script:** `scripts/setup_argos.py`

**Public API:**
```python
class Translator:
    def __init__(self):
        """Warm up all 4 language pairs (en→it, es, fr, de) on construction."""
    def translate(self, word: str, target: str) -> str:
        """Translate an uppercase English word to target language code
        (one of: it, es, fr, en, de). Returns the translated string."""
```

### Decisions

1. **Library: `argostranslate` (Python package).** Used directly, version
   pinned in `pyproject.toml`. No GUI / CLI tooling pulled in.

2. **Offline pack installation.** All 4 EN→target packs (`.argosmodel`
   files for IT, ES, FR, DE) are pre-installed at provisioning time by
   `scripts/setup_argos.py`. The runtime never attempts to fetch a pack,
   so missing-pack equals broken setup, not a recoverable runtime error.

3. **Startup warmup.** `Translator.__init__()` triggers one dummy
   translation per pair. Argos lazy-loads on first use; warmup moves
   that latency from "the user's first word after boot" to startup.
   Memory cost (~few hundred MB resident) is acceptable on Pi 5 8 GB.

4. **Casing.** Recognizer emits uppercase ("HELLO"). MT models translate
   lowercase more accurately. `translate()` lowercases before sending to
   Argos and returns the lowercase result. The TTS pronounces the same
   either way.

5. **LRU cache.** `translate()` is wrapped with
   `functools.lru_cache(maxsize=128)`. Repeated words within a session
   translate once. Tiny memory cost.

6. **Identity short-circuit.** When `target == "en"` we return the
   lowercased input unchanged. No Argos call.

7. **Failure handling.** Any exception from Argos (missing pack,
   internal error) is caught: log a warning, return the lowercased
   original word. The worker never crashes. Empty input → return empty
   string silently.

8. **Worker / queue structure.** A separate process reads from
   `translation_in_queue: (word, target_lang)` and writes to
   `tts_in_queue: translated_string`. Each input message carries its
   own `target_lang`, so language changes propagate without restarting
   the worker or sharing state.

9. **Performance target.** 100–500 ms per word on Pi 5. If profiling
   shows worse, swap that pair's model for a smaller Argos variant.

### Out of scope for this sub-feature

- Sentence-level translation (PLAN.md keeps v1 word-by-word).
- Other source languages (we always source EN — fingerspelling produces
  English letters).
- Re-translation on language change (per #8, language is per-message).
- Custom MT models (we trust the upstream Argos packs).

---

## #6 — Text-to-speech (Piper) ✅ LOCKED

**Module:** `src/asl_live/tts/speaker.py`
**Voice install path:** `/opt/piper/voices/<lang>.onnx` (+ `.json` config)

**Public API:**
```python
class Speaker:
    def __init__(self):
        """Preload all 5 voices."""
    def speak(self, text: str, lang: str) -> None:
        """Synthesize and play. Blocks until playback finishes."""
```

### Decisions

1. **Library: `piper-tts` (Python package).** Version pinned in
   `pyproject.toml`. One voice per language: 5 voices total in
   `/opt/piper/voices/` (IT, ES, FR, EN, DE). Specific voice models
   chosen by ear during phase-7 provisioning, defaulting to each
   language's medium-quality option.

2. **Preloading.** All 5 voices loaded into memory at `Speaker.__init__()`
   so language switches are instant. Each voice ~50–100 MB; total
   resident is fine on Pi 5 8 GB.

3. **Audio output: ALSA → USB speaker.** ALSA's default device is
   pointed at the USB speaker via `/etc/asound.conf` during phase-7
   provisioning. Piper outputs raw PCM (typically 16-bit, 22 050 Hz);
   `sounddevice.play()` hands that to ALSA.

4. **Backpressure: bounded queue, drop-oldest.** The TTS input queue is
   size 3. If a 4th word arrives while we're still speaking, the oldest
   queued word is dropped. Rationale: keeps audio close to real-time
   when the user signs faster than speech; unbounded queueing would
   produce a growing lag the user can't recover from. The trade-off
   (occasional dropped words) is documented in user-facing docs.

5. **Failure handling.**
   - Missing voice file → log a warning, fall back to **eSpeak-NG** for
     that language only (eSpeak-NG is tiny, ships with Linux, much
     lower quality but always works).
   - Synthesis exception → log, skip that word, do not crash the
     worker.
   - Empty input → no-op.

6. **Worker structure.** Mirror of the Translator worker: blocking loop
   reading `tts_in_queue: (text, lang)`. Each message carries its own
   `lang`, so language changes propagate without restarting the worker
   or sharing state. `speak()` blocks until ALSA finishes playing.

7. **Boot announcement.** On startup, `lang_menu.py` (sub-feature #9)
   calls `speak(<language_name>, current_lang)` so the user hears the
   active language and knows the device is ready. Only "device ready"
   signal in the absence of a screen.

8. **Performance target.** Piper synthesizes ~5–10× real-time on Pi 5
   (~100–200 ms for a 1-second utterance). End-to-end latency from
   SPACE-sign commit to first audio: translation ~300 ms + synthesis
   ~200 ms ≈ 500 ms. Comfortable for the UX.

### Out of scope for this sub-feature

- Voice cloning, custom voices, prosody control.
- Audio file output (no save-to-WAV mode).
- Bluetooth speakers (USB / 3.5 mm only in v1).
- Speech-rate or volume controls (system ALSA mixer if needed).
- Mid-utterance interruption (a new message can't cut off the current
  one — it lands in the bounded queue or gets dropped).

---

## #7 — LCD display ✅ LOCKED

**Module:** `src/asl_live/ui/lcd.py`
**Hardware:** DFRobot DFR0063 — 16×2 character LCD with PCF8574 I2C
backpack, on Pi I2C bus 1 (GPIO 2 = SDA, GPIO 3 = SCL).

**Public API:**
```python
class LCDWriter:
    def update(self, word: str, lang: str, status: str) -> None:
        """Render line 1 = word (last 16 chars), line 2 = `<lang>|<status>`.
        Diffs against the previous content and only writes changed cells."""
```

### Decisions

1. **Library: `RPLCD`** (HD44780 + PCF8574 wrapper). Pinned in
   `pyproject.toml` under the `[pi]` install profile — does not install
   on the dev PC.

2. **I2C address auto-detect.** DFR0063 typically replies on `0x27`,
   some clones on `0x3F`. On startup probe both; use whichever
   responds. If neither responds, log an error and continue running
   without a screen (degraded mode, not a crash).

3. **Cell-diffing update strategy.** `LCDWriter` keeps a 32-character
   shadow of the previous render and only writes cells whose content
   has changed. Eliminates the flicker that comes from clear-and-rewrite.

4. **Single update entrypoint.** `update(word, lang, status)` is the
   only public method. No separate line-writes, no flush/commit calls
   — the diff lives inside the writer.

5. **Word truncation.** Words longer than 16 chars: show the **last**
   16 (most useful while typing). When the word commits and the buffer
   clears, line 1 clears.

6. **Line-2 format:** `<LANG>|<STATUS>` left-justified, padded to 16
   chars. Status codes:
   - `IDLE` — device on, not capturing.
   - `REC ` — capturing letters.
   - `TX  ` — translating.
   - `TTS ` — speaking.
   - `LANG` — language menu open (line 1 shows the candidate language
     name in this state).

7. **Backlight always on.** ~0.05 W draw is negligible; a dark display
   reads as "device broken." Toggleable via `RPLCD` if a future power
   mode demands it.

8. **Update source.** A small thread in the main process subscribes to
   recognizer commit events and lifecycle state changes and calls
   `update()`. Not run from the recognizer worker (kept pure / fast).
   Updates fire on state change only, never per-frame.

9. **Boot splash.** First render after boot: line 1 = `ASL-live`,
   line 2 = `loading...`, while Piper / Argos / camera initialize.
   Replaced by `<lang>|IDLE` once startup is complete.

10. **Failure handling.** Mid-session I2C write failures (loose wire,
    flaky cable) are logged and swallowed. The next `update()` call
    re-attempts. The system keeps running without the screen; the user
    just loses visual confirmation.

11. **Testability split.** Pure formatter
    `format_lines(word, lang, status) -> tuple[str, str]` is
    unit-tested without hardware (truncation, padding, status-code
    formatting). The I2C writer has only a smoke test on the Pi.

### Out of scope for this sub-feature

- Custom 5×8 glyphs / icons (RPLCD supports them — not needed in v1).
- Scrolling long words across line 1 (we show the last 16 chars
  instead).
- Brightness control / dimming (no PWM on the backlight pin via
  PCF8574).
- Multi-line wrap of the word across lines 1 and 2 (line 2 is
  status-only).

---

## #8 — Buttons ✅ LOCKED

**Module:** `src/asl_live/ui/buttons.py`
**Hardware:** 2 momentary tactile pushbuttons, each wired GPIO pin → GND.

**Public API:**
```python
class Buttons:
    def __init__(self, on_b1: Callable[[], None], on_b2: Callable[[], None]):
        """Bind callbacks to the two physical buttons. Callbacks fire on
        the gpiozero callback thread — handlers should be quick and
        thread-safe (typically just enqueue an event)."""
```

### Decisions

1. **Library: `gpiozero`.** Pinned under the `[pi]` install profile.
   Provides edge detection, debounce, and callback threading.

2. **Pin assignment** (locked, drives wiring docs):
   - **B1 (main)** → GPIO 17 (header pin 11)
   - **B2 (aux)** → GPIO 27 (header pin 13)
   Adjacent on the header for easy breadboard wiring. Clear of I2C
   (GPIO 2/3) and UART (GPIO 14/15).

3. **Topology.** Each button has one leg on its GPIO pin and one leg on
   GND. `gpiozero.Button(pin, pull_up=True)` enables the Pi's internal
   pull-up resistor — pressed reads LOW. No external resistors.

4. **Hardware debounce.** `bounce_time=0.03` (30 ms) on each Button.
   Filters mechanical bounce typical of cheap tactile switches.

5. **Short-press only.** No long-press detection. The 2-button + audio
   menu UX (sub-feature #9) doesn't need it — B2 short-press is itself
   the language-menu trigger when idle. Simpler, fewer accidental
   triggers.

6. **Callback thread → event queue.** `gpiozero` invokes callbacks on
   its internal thread. The bound callbacks do nothing more than push
   `ButtonEvent("B1")` or `ButtonEvent("B2")` onto the orchestrator's
   event queue and return. All state-machine logic runs on the main
   thread.

7. **Failure handling.** GPIO init exception (permission, hardware) is
   logged and swallowed; the application continues running without
   button input. Useful when developing on the PC where there is no
   GPIO at all.

8. **PC keyboard fallback** (`--keyboard-buttons` dev flag). On the
   development PC, keys `1` and `2` invoke the same callbacks as B1/B2.
   Used by the phase-3 PC demo. Not enabled in production Pi runs.

9. **Testability.** This module is intentionally thin — the
   interesting logic lives in the orchestrator state machine
   (sub-feature #10) and the audio language menu (#9), both of which
   are tested by enqueueing `ButtonEvent` objects directly.

### Out of scope for this sub-feature

- Long-press, double-press, or chorded press detection.
- Hardware debounce circuitry (gpiozero's software debounce is enough).
- Hot-pluggable buttons / dynamic re-binding.
- More than 2 buttons (UX is locked at 2 by discussion §8).

---

## #9 — Audio language menu ✅ LOCKED

**Module:** `src/asl_live/ui/lang_menu.py`
**Persistence:** `~/.aslive/config.json`

**Public API:**
```python
class LanguageMenu:
    def __init__(self, speaker: Speaker, lcd: LCDWriter, config_path: Path): ...
    def open(self) -> None:           """Enter menu, announce current candidate."""
    def on_b2_press(self) -> None:    """Cycle to next candidate."""
    def on_b1_press(self) -> None:    """Cancel, restore previous selection."""
    def tick(self) -> None:           """Called periodically; auto-confirms on 3 s timeout."""
    @property
    def is_open(self) -> bool: ...
    @property
    def current_lang(self) -> str: ...
```

### Decisions

1. **Persistence: `~/.aslive/config.json`.** Schema:
   ```json
   {"target_lang": "it", "version": 1}
   ```
   Loaded on boot, written atomically (write-to-temp + rename) on
   confirm. `version` field is forward-compatibility insurance.

2. **First-boot default: English (`"en"`).** Neutral choice when no
   config file exists. The user changes it via the menu on first use
   and the file is created.

3. **Cycle order:** IT → ES → FR → EN → DE → IT.

4. **Announcement strategy: each language speaks its own name in its
   own voice.**
   - IT candidate → Piper Italian voice says "Italiano".
   - ES → Spanish voice says "Español".
   - FR → French voice says "Français".
   - EN → English voice says "English".
   - DE → German voice says "Deutsch".

   On confirm, "OK <lang>" is spoken in the *newly chosen* language's
   voice. On cancel, the *previous (preserved)* language's name is
   spoken in its voice — the user always hears something when leaving
   the menu, removing ambiguity about which selection ended up active.

5. **Boot announcement.** After all workers warm up, the persisted
   language's name is spoken in its own voice. Only "device ready" cue
   in the absence of a screen.

6. **Confirm timeout: 3 s.** Each candidate announcement resets the
   timer. After 3 s of no B2 press, the current candidate is confirmed
   and persisted. Walk-away behavior is acceptable: if the user
   accidentally cycled before walking off, they re-open and fix it.

7. **State transitions:**
   - **closed** + B2 → enter menu, announce current candidate, start
     timer.
   - **open** + B2 → advance candidate, re-announce, restart timer.
   - **open** + B1 → cancel, restore previous, announce previous,
     close.
   - **open** + 3 s elapsed → confirm, persist, announce "OK <lang>",
     close.

8. **LCD during MENU_OPEN.**
   - Line 1: candidate language name (e.g., `Italiano`).
   - Line 2: `LANG|<code>` (e.g., `LANG|IT  `).
   On exit, the LCD reverts to whatever idle/capture render the
   orchestrator drives next.

9. **Routing.** When `is_open` is True, the orchestrator delivers
   button events to `on_b1_press` / `on_b2_press`; capture-mode
   handlers see nothing. When `is_open` is False, the orchestrator
   handles capture-mode logic and only `open()` is callable from
   outside.

10. **Testability.** Unit tests mock `Speaker.speak()` and inject a
    fake clock for the timeout. No GPIO required. Cover: cycle order,
    confirm-on-timeout, cancel-restores-previous,
    persistence-write-atomic-on-confirm, persistence-not-written-on-cancel.

### Out of scope for this sub-feature

- Visual menu on the LCD beyond the candidate name (no scrolling list,
  no arrow indicators).
- Voice prompts for menu help / instructions.
- Adding / removing languages at runtime (set is fixed at IT/ES/FR/EN/DE).
- Per-user profiles.

---

## #10 — Process orchestration ✅ LOCKED

**Module:** `src/asl_live/pipeline/main.py`
**Entry point:** `python -m asl_live.pipeline.main` (dev) or
`asl-live.service` systemd unit (production).

### Decisions

1. **4 processes total.**
   - **Main** — orchestrator state machine, LCD update thread, menu
     `tick()` thread, signal handlers.
   - **Recognizer** — camera capture, MediaPipe, classifier, debouncer.
   - **Translator** — Argos.
   - **TTS** — Piper synthesis + ALSA playback.

   `multiprocessing.Process` for the three workers. CPU-heavy work
   isolated from the GIL.

2. **Queues** (all `multiprocessing.Queue` except where noted):

   | Queue | Direction | Payload | Bound | Overflow |
   |---|---|---|---|---|
   | `recognizer_out_queue` | recognizer → main | `CommitEvent` | 16 | block (events are rare) |
   | `translation_in_queue` | main → translator | `(word, target_lang)` | 16 | block |
   | `tts_in_queue` | translator → tts | `(text, lang)` | **3** | **drop-oldest** (per #6) |
   | `button_event_queue` | gpiozero thread → main | `ButtonEvent` | 16 | block |

   The button queue is `queue.Queue` (in-process) since gpiozero runs
   in the same process as main.

3. **Lifecycle state machine** (main thread):
   - **IDLE** — device on, not capturing. B1 → CAPTURING. B2 → open
     language menu (state becomes MENU until menu closes).
   - **CAPTURING** — recognizer active. B1 → IDLE (cancels current
     word). B2 → force-send current word, remain CAPTURING.
   - **MENU** — `LanguageMenu.is_open`. Button events routed to the
     menu; recognizer events ignored.

4. **Recognizer start/stop.** A `multiprocessing.Event` named
   `capture_enabled` gates the recognizer's main loop. Main sets it on
   IDLE→CAPTURING and clears it on CAPTURING→IDLE. While cleared, the
   recognizer waits on the Event — no MediaPipe work, no CPU burn.

5. **Word buffer lives in main, not the recognizer.**
   - On `LETTER` event → append to buffer, refresh LCD.
   - On `DELETE` event → pop last char (no-op on empty), refresh LCD.
   - On `SPACE` event → if buffer non-empty, push to
     `translation_in_queue`, clear buffer, set LCD status to `TX`;
     empty buffer → silent no-op.
   The recognizer just emits committed gestures; UX semantics are main's
   concern.

6. **Logging.** `logging` to a rotating file:
   - `/var/log/asl-live/app.log` when running under systemd.
   - `~/.aslive/logs/app.log` otherwise.
   Level INFO by default, DEBUG with `--verbose`. Each worker tags log
   records with its process name.

7. **Crash policy: log, don't auto-restart (v1).** Main checks each
   worker's `is_alive()` once per second. On a worker exit it logs the
   failure and continues running with degraded functionality. Restart
   with state recovery is a phase-7 item if real-world failures
   warrant it.

8. **Graceful shutdown** on SIGINT (Ctrl+C) and SIGTERM (systemd stop):
   - Set `shutdown_event = multiprocessing.Event()`.
   - Push sentinel `None` to each worker input queue.
   - Workers detect either signal, drain in-flight work, exit.
   - Main joins each worker with a 3 s timeout, then `terminate()` if
     still alive.
   - LCD shows `shutdown...`, backlight off on final exit.

9. **Config loading at startup.** `load_config()` reads
   `~/.aslive/config.json` (creates with defaults if missing) and
   exposes a typed `Config` object (pydantic) used by all components.
   Includes `target_lang`, debounce thresholds (`STABLE_FRAMES`,
   `GAP_FRAMES`, `MIN_CONF`), GPIO pin assignments, and log level.

10. **Testability.** Each worker is testable in isolation by feeding
    its input queue directly. The main state machine is testable by
    injecting a fake `button_event_queue` and a fake
    `recognizer_out_queue`, then asserting on calls to mocked
    `LCDWriter`, `Translator`, `Speaker`. No camera, no GPIO, no
    audio device required for unit tests.

### Out of scope for this sub-feature

- Multi-tenant operation (one device, one user, one session at a time).
- Hot-reload of config (changes require restart).
- Auto-restart of crashed workers (deferred to phase 7 if needed).
- IPC mechanisms other than stdlib queues (no ZeroMQ, no shared
  memory).
- Web / network admin interface (offline device, no exposed services).
