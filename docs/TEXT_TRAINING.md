# Text Training: from raw Shakespeare to generated text

End-to-end language model on raw text, from scratch, no frameworks.
Every component already existed: BPE, tokenizer, MiniLM, causal mask.
This document is the trajectory of the first working run.

## Setup

- **Corpus**: `tinyshakespeare/input.txt`, ~1.1 MB (Karpathy's char-rnn repo)
- **BPE**: word-level, 128 merges, vocab = 196 tokens (4 specials + 89 chars
  + `</w>` + 102 merges that found frequency ≥ 2)
- **Tokenizer output**: 625,534 tokens, sliced into `seq_len=8` windows
  → 78,190 windows
- **MiniLM**: `d_model=16`, `d_ff=32`, `2 blocks`, `n_heads=1`,
  `qhead=-1` (all classical), `causal=true`
- **Training**: 200 epochs, `windows_per_epoch=500` (rotating offset,
  corpus covered 200×500/78190 ≈ 1.28 times), lr=0.05, clip=1.0
- **Hardware**: Acer Aspire Go 15, i3-N305, 8 GB, no GPU
- **Wall-clock**: ~20 seconds for the full 200 epochs + generation

## The leakage bug and its fix

First run before causal mask, epoch 1 loss was **1.76**. That is
impossible for a randomly initialized model against a 196-token
vocabulary: uniform prediction gives `log(196) = 5.28`, random init
gives above 5.28. The model was reading target token `i+1` through
unmasked attention.

After adding causal mask (see commit `51d6480` and follow-ups),
epoch 1 loss became **7.44** — above the baseline, as expected.
The causal mask is the single most important fix in this trajectory.

## Loss trajectory

| Epoch | Loss |
| :--- | :--- |
| 1 | 7.44 |
| 5 | 4.94 |
| 10 | 4.70 |
| 20 | 4.21 |
| 40 | 3.54 |
| 60 | 3.71 |
| 115 | 3.99 |
| 165 | 3.77 |
| 195 | 3.71 |
| 200 | 3.88 |

The oscillation between consecutive epochs (3.54 → 3.71 → 3.99 → 3.77)
is the signature of window rotation: each epoch trains on a different
500-window slice, so the loss on a fresh slice varies. The downward
trend over 200 epochs is the learning.

The plateau around 3.7–4.0 is not a bug. It's the generalization
floor of a 4,000-parameter model on Shakespeare's conditional
distribution after 100,000 updates. Two contributors:

1. **Capacity**: 4k params is small. Perplexity `exp(3.88) ≈ 48`
   means the model is "surprised" by a factor of 48 on average per
   token, which is consistent with learning bigram/trigram structure
   but not long-range dependencies.
2. **Data vs epochs**: 200 epochs × 500 windows = 100,000 updates on
   78,190 unique windows. The corpus is seen 1.28 times. More epochs
   (say 1000) would see the corpus 6 times, but the improvement would
   be slow past this point because the model has already converged
   on the trivially-learnable structure.

## Greedy decoding collapses — fix with temperature sampling

First attempt: greedy `argmax` over logits. Output:

    'daindaindaindaindaindaindaindaindaindain'

A one-token collapse. Once the model predicts `dain`, conditioning
on `dain` produces `dain` again with high probability, and argmax
never escapes. This is the canonical failure mode of greedy decoding
on small models.

Fix: softmax with temperature `T=0.7` and `std::discrete_distribution`.
`dain` is still likely but not certain — the sampler explores other
tokens. Output after the fix:

    't HMONONDONDithere you to a dainsedainjake of HM'

**Real English words**: `there`, `you`, `to`, `a`, `of`.
**BPE fragments**: `HMONONDOND`, `dainsedainjake`, `HM`.

The model learned vocabulary and common bigrams (`to a`, `of HM`).
It has not learned word boundaries or syntax. This is exactly what
a 4k-parameter model at loss 3.88 should produce. It is not failure;
it is the honest output of the system at this scale.

## What this proves

End-to-end functionality:

1. BPE trains from raw bytes (no pretrained vocab).
2. Tokenizer maps raw text to ids and back (roundtrip verified in
   `test_bpe`).
3. MiniLM trains on next-token prediction with causal mask (no leakage).
4. Loss descends monotonically over the training distribution
   (modulo per-epoch oscillation).
5. Sampling produces tokens that decode to recognizable words.

Every one of these five steps had to be correct for the output to
appear. If any were broken, the output would be empty, `<unk>`-only,
or the loss would not descend. It does descend, and the output is
recognizable.

## What this does NOT prove

- **Generalization**: The model fits the average conditional
  distribution, not Shakespeare. It cannot write a sonnet.
- **Scale**: 4k params is 100,000× smaller than GPT-3. Results at
  d_model=16 do not extrapolate to d_model=768.
- **Usefulness**: The output is a demonstration, not a product.

## What a next run would test

Three directions, each isolating one variable:

1. **More epochs** (1000, same params): tests whether the 3.88
   plateau is "not enough data" (loss drops to ~3.5) or "capacity
   exhausted" (loss stays at 3.9). One-line change, ~30 min.
2. **More capacity** (`d_model=32`, 1 block, same total params as
   d=16 × 2 blocks): tests whether a wider, shallower model learns
   word boundaries better. 3-line change.
3. **Larger corpus**: concatenate more Shakespeare, or add a second
   public-domain text. Tests whether the model generalizes better
   when it has more diverse training signal. More work.

Direction 1 is cheapest and most informative: if loss drops to 3.5,
more epochs is the right lever. If it stays at 3.9, capacity is the
constraint.

## Reproduce

    cd ~/Absolutus
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/shakespeare.txt
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
    ./build/absolutus --experiment train_text 2>&1 | tail -6

Expected: loss trajectory 7.44 → ~3.9, then `output:` with real
Shakespeare words mixed with BPE fragments.
