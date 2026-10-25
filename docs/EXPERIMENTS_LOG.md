# Experiments Log

Quantitative record of training experiments. Each entry states the
hypothesis being tested, the configuration, the measured outcome, and
the conclusion drawn.

---

## Baseline (reference)

- **Config**: 200 epochs, d_model=16, d_ff=32, 2 blocks, 128 merges
- **Loss**: 7.44 (ep 1) -> 3.88 (ep 200), plateau ~3.7-4.0 after ep 100
- **Output**: `'t HMONONDONDithere you to a dainsedainjake of HM'`
- **Reading**: end-to-end functional, but capacity-limited (fragments
  dominate, no full words except short ones like `you`, `to`, `a`)

## Experiment A: more data (1000 epochs, same capacity)

- **Hypothesis**: 200 epochs see the corpus 1.28 times. More epochs
  should let the model consolidate word boundaries.
- **Config**: 1000 epochs, d_model=16, d_ff=32, 2 blocks, ~4k params
- **Loss**: 7.44 (ep 1) -> **3.754 (ep 1000)**, delta = -0.13 vs baseline
- **Output**: `'. GREGREGREGRESON: AMPSAMPSAuued'`

**Reading**: Loss improves by 3.3% but output quality degrades.
`GREGREGREG` (GREGORY, a Shakespeare character) and `AMPSAMPS` exhibit
the same repetition-collapse failure as the pre-temperature-sampling
run, just on different attractors than the baseline's `daindain`.
More epochs makes the model MORE confident in whatever pattern it has
locked onto, not more accurate. The corpus was seen 6.4 times by
epoch 1000; this is overfitting to conditional-structure biases that
the 4k-param model cannot resolve into word-level features.

## Experiment B: more capacity (d_model=32, 1 block, 200 epochs)

- **Hypothesis**: The plateau is capacity-limited. A wider, shallower
  model has more representational room even with similar parameter
  count, so it should learn word boundaries.
- **Config**: 200 epochs, d_model=32, d_ff=32 (unchanged), 1 block,
  ~4k params (same as baseline)
- **Loss**: 5.44 (ep 1, lower than 7.44 because wider init is closer to
  uniform) -> **3.849 (ep 200)**, delta = -0.03 vs baseline
- **Output**: `'UD RICHARDULANograckaleacaining comelead'`

**Reading**: Loss is essentially identical to the baseline (3.85 vs
3.88) but output is qualitatively much better:
- `RICHARD` — a full Shakespeare character name (5 chars, 2-3 BPE pieces)
- `come`, `lead` — real English words
- No immediate repetition collapse

The model has consolidated longer tokens into coherent units, which
d=16 cannot represent. This happens without lowering the aggregate
per-token cross-entropy — the loss plateau is not the right metric
for output quality at this scale.

## The key finding: loss and output quality diverge

| Config | Params | Epochs | Loss | Output |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | ~4k | 200 | 3.88 | fragments + short words |
| A: 1000 ep | ~4k | 1000 | **3.75** | fragments + repetition collapse |
| B: d=32 | ~4k | 200 | 3.85 | **real names + words** |

A has the lowest loss but the worst output. B has loss essentially
equal to the baseline but the best output. **The aggregate
cross-entropy on 196 tokens does not distinguish between "learns
correct word boundaries" and "learns fragment-collocation with high
confidence."** Both reduce the metric; only the first produces
coherent text.

Practical implication: at this scale, tuning for loss is
counterproductive. The right signal is a qualitative assessment of
the generated output. This is a small-model regime where perplexity
and usefulness have decoupled — which is known in the literature
(Brown et al. 2020 on GPT-3 showed this starts happening below
certain scale thresholds) but seeing it directly is instructive.

## What neither experiment escapes

- **Loss plateau**: 3.75-3.88 regardless of configuration. The model
  has converged on the conditional distribution it can represent.
- **Token-level quality**: neither produces grammatical text. A
  sonnet is not happening at d=32.
- **The corpus size**: 1.1 MB is small. Both experiments exhaust the
  learnable structure, just in different ways.

## What this suggests for next steps

The direction that actually addresses the finding:

1. **Not more epochs**. A shows this is counterproductive at fixed
   capacity.
2. **Not just more d_model**. B is better but the effect is per-token
   representation, not deeper architecture.
3. **Better training signal**: at this scale, what would actually
   help is more data diversity (second corpus), not more of the same.
   The model has extracted everything it can from Shakespeare at this
   capacity.

Concrete next experiment (not run): same d_model=32, but with
sequence-level loss instead of per-token CE, or with a small amount
of labeled contrast between "real word boundary" and "wrong boundary".
This is out of scope for today.

## Reproduction

    cd ~/Absolutus
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/shakespeare.txt
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)

    # Baseline
    ./build/absolutus --experiment train_text

    # Experiment A
    ./build/absolutus --experiment train_text --lm-epochs 1000

    # Experiment B
    ./build/absolutus --experiment train_text --lm-epochs 200 --d-model 32 --blocks 1
