# Experiments Log

Quantitative record of training experiments. Each entry states the
hypothesis being tested, the configuration, the measured outcome, and
the conclusion drawn.

**Methodological note.** Every conclusion below is based on 100-token
samples, not 30-token. The 30-token samples that informed an earlier
version of this document were noisy: individual draws over/under-
represented specific token attractors. The 100-token samples give a
representative view of the model's distribution (albeit still one
draw — multi-seed ablation is listed as future work).

**Sampling note.** All outputs below use top-k=40 + repetition
penalty=1.3 (window 20) + temperature=0.7. Without these, all three
configs collapse into repetition (see Sampling Ablation section).

---

## Baseline (reference)

- **Config**: 200 epochs, d_model=16, d_ff=32, 2 blocks, 128 merges,
  ~4k params
- **Loss**: 7.44 (ep 1) -> 3.88 (ep 200)
- **Output (100 tok)**: `, ADERICHis uput I have a p Goven thy of wrood
  HMONDERICHMOLAThat was RBated a boouslack up MOND HARDInhim our
  stower hould a shast the dake of LONDertomarkne`
- **Reading**: strongest local phrase quality of the three.
  `I have a` (full English trigram), `thy of`, `the dake of LOND...`
  (the "Duke of [city]" pattern, very Shakespearean). But no
  higher-order structure — no dialogue format, no consistent
  capitalization pattern. Fragments dominate between phrases.

## Experiment A: more data (1000 epochs, same capacity)

- **Hypothesis**: 200 epochs see the corpus 1.28 times. More epochs
  should consolidate word boundaries.
- **Config**: 1000 epochs, d_model=16, d_ff=32, 2 blocks, ~4k params
- **Loss**: 7.44 (ep 1) -> **3.754 (ep 1000)** — delta -0.13 vs baseline
- **Output (100 tok)**: `. GRESON: MPSAMon, thuely them ir mbrokn the
  pot. SAMPREGON: and aince with theet I pir, Thuin. SAMPSOREGRY: I
  will sh the pent them uin, they guloir. SON: A`
- **Reading**: strongest structural learning of the three. The
  `NAME: dialogue` pattern appears 4 times. `GRESON`, `SAMPREGON`,
  `SAMPSOREGRY`, `SON` are corrupted variants of `GREGORY`,
  `SAMPSON`, `SAMPSON`, `SON` — actual character labels in Romeo
  and Juliet. The model learned the play format: capitalized word,
  colon, lowercase text. Also has coherent phrases (`I will`,
  `and with`). Word-level quality is worse than baseline
  (`GRESON` vs `Goven`), but higher-order structure is far better.

## Experiment B: more capacity (d_model=32, 1 block, 200 epochs)

- **Hypothesis**: The plateau is capacity-limited. A wider, shallower
  model has more representational room even with similar parameter
  count.
- **Config**: 200 epochs, d_model=32, d_ff=32 (unchanged), 1 block,
  ~4k params
- **Loss**: 5.44 (ep 1, lower because wider init is closer to uniform)
  -> **3.849 (ep 200)** — delta -0.03 vs baseline
- **Output (100 tok)**: `aving steacir hongainesturdy HMONDUWhat AThe
  prothochich alefhastare wear RICHMONDADey my talardeashath longost
  of HARDONERICHMy thould what and Loo how wain`
- **Reading**: intermediate. Has `RICHMOND` spelled correctly (a
  Henry VI character that neither A nor baseline spells correctly),
  has `The X Y wear` (syntactic), has `My ... what and ... how ...`
  (attempt at structure). But fragments dominate, no dialogue
  format, no consistent capitalization pattern like A. Word-level
  quality between baseline and A.

## The key finding: loss does not rank output quality

| Config | Epochs | Loss | Primary strength |
| :--- | :--- | :--- | :--- |
| Baseline | 200 | 3.88 | local phrases (`I have a`, `the dake of`) |
| A: 1000 ep | 1000 | **3.75** | **higher-order structure** (dialogue format) |
| B: d=32 | 200 | 3.85 | some full-word spellings (`RICHMOND`) |

A has the lowest loss and the most structural learning. Baseline
has the highest loss and the best phrase-level quality. B is between
on all axes. **None of these correlations is causal**: the loss
metric compresses all of these effects into a single average.

The 30-token sample earlier suggested B was best. With 100 tokens,
that conclusion collapses. A single short sample is not a measure
of model quality.

---

## Sampling Ablation

Same models, different sampling strategies, evaluating the frequency
of repetition collapse.

### Greedy (argmax, first attempt)

output: 'daindaindaindaindaindaindaindaindaindain'

Total collapse. Once the model predicts `dain`, argmax re-predicts it
forever. This is not a model bug — argmax is a deterministic map from
context to next token, and small models have absorbing states.

### Temperature sampling (T=0.7, no top-k)

output: 't HMONONDONDithere you to a dainsedainjake of HM'

Collapse broken — variety present. Real words: `there`, `you`,
`to`, `a`, `of`. But fragment repetition (`HMONON`) and BPE
gluing (`dainsedainjake`) remain.

### Top-k=40 + repetition penalty=1.3 (window 20)

output: ', ADERICHis uput I have a p Goven thy of wrood HMONDERICHMOLAThat...'

Fragments reduced. Real phrases (`I have a`), real structural
patterns (`the dake of`). Repetition gone.

### The lesson

The apparent "quality" of a small model is a **joint** property of
the model weights and the sampling strategy. The 1.76 -> 7.03 loss
fix (causal mask) was necessary but not sufficient for readable
output. Top-k was also necessary. Many deployment failures attributed
to "small model is bad" are actually "small model plus greedy decoding
is bad".

---

## Conclusion

1. **The plateau is real but not monotonic across metrics.** A
   improves loss by 3.3% and structural output; baseline keeps better
   phrase-level output at higher loss. No single loss number
   summarizes model quality at this scale.

2. **Top-k sampling is essential**, not optional. All three configs
   collapse under greedy decoding. The same models under top-k produce
   output that reveals their different strengths.

3. **Single short samples are noisy.** The 30-token sample that
   suggested "B is best" was overturned by a 100-token sample. Future
   experiments should use at least 100 tokens and ideally multiple
   seeds.

## Reproduce

    cd ~/Absolutus
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/shakespeare.txt
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)

    # Baseline with top-k
    ./build/absolutus --experiment train_text --samples 100

    # Experiment A
    ./build/absolutus --experiment train_text --lm-epochs 1000 --samples 100

    # Experiment B
    ./build/absolutus --experiment train_text --lm-epochs 200 --d-model 32 --blocks 1 --samples 100

## Future work

- Multi-seed evaluation (5+ seeds per config, aggregate statistics)
- Sentence-level BLEU or held-out perplexity per config
- Corpus diversity test (second public-domain text)
- Word-boundary-aware loss (distinguish `GRESON` from `GREGORY`)
