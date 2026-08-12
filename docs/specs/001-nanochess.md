# Spec 001 — nanochess

Status: approved
Date: 2026-08-12

## Problem

Train a decoder-only transformer from scratch on chess games in PGN notation, so
that it predicts the next move. The model is not trying to play strong chess.
The question it answers is narrower and objectively measurable:

> Can a small language model, shown nothing but sequences of moves, infer the
> rules and state of chess well enough to emit **legal** moves?

That question has a number attached to it, which is the point. Most portfolio
LLM projects report perplexity, which is uninterpretable to a reader. This one
reports **legal-move rate**: the percentage of sampled moves that are legal in
the actual board position, checked against a real chess engine's move generator.

## Hard constraints

| Constraint | Value |
| --- | --- |
| Hardware | MacBook (Apple Silicon, MPS); optional free Colab T4; must also run on CPU |
| Full training run | < 4 hours |
| Parameters | single-digit millions to ~20M |
| Dependencies | torch, python-chess, zstandard. No Trainer, no Lightning, no wandb |
| Written by us | tokenizer, model, training loop, sampler, eval harness |
| Not written by us | autograd, AdamW |

## Design decisions

### D1 — Tokenizer: move-level over UCI

**Decision:** one token per ply, using UCI move notation (`e2e4`, `g1f3`,
`e7e8q`). Vocabulary is a fixed enumeration of every geometrically possible UCI
move — 1968 of them — plus 4 special tokens, for a vocab of **1972**.

The enumeration is generated, not scraped from the corpus: for each of the 64
origin squares, place a lone queen and a lone knight on an empty board and take
the union of their destinations (this covers every sliding, knight and king move,
and castling, which in UCI is just `e1g1`). Then add the 176 promotion moves
(8 files × up-to-3 destination files × 4 promotion pieces × 2 colours). A
corpus-derived vocab was rejected because it would make the tokenizer's
`<unk>` rate a function of which month of Lichess you downloaded.

**Rationale.** The eval is the centrepiece, and the eval is only clean if one
token equals one move. With UCI, checking a generated token is
`chess.Move.from_uci(tok) in board.legal_moves` — a single unambiguous test.
The rejected alternatives:

- *Character-level over SAN* — maximum from-scratch purity and a tiny embedding
  table, but a game becomes ~500 characters instead of ~80 tokens, so the same
  board coverage costs 6× the context. Worse, it introduces a second failure
  mode (malformed strings that do not parse at all), which splits the headline
  metric into "% well-formed" and "% legal" and muddies the one number the
  README leads with.
- *Small BPE over SAN* — the most modern-looking option and the worst fit.
  Token boundaries do not align with move boundaries, so "this token was
  illegal" stops being a well-defined statement.
- *SAN move-level* — SAN is board-dependent (`Nf3` means different moves in
  different positions) and its vocabulary is unbounded in principle. UCI is
  position-independent and closed.

**Acknowledged weakness:** a move-level tokenizer is close to a dictionary
lookup, so "we wrote our own tokenizer" is a weak claim on its own. Mitigation:
ship a second implementation — a character-level tokenizer over the same
corpus — behind the same `Tokenizer` protocol, round-trip tested, so the
abstraction is real and the choice is demonstrated rather than asserted.

### D2 — The model sees moves only, never the board

No FEN conditioning, no piece-placement features, no board planes. The model
gets a flat sequence of move tokens and must infer the position from the move
history alone. Feeding it the board would inflate the legal-move rate and
destroy the meaning of the measurement.

### D3 — Architecture

Decoder-only pre-norm transformer, written from scratch. No `nn.Transformer`,
no `nn.MultiheadAttention`.

- learned token embeddings + learned positional embeddings
- N × [ LayerNorm → causal multi-head self-attention → residual;
        LayerNorm → MLP (4× expansion, GELU) → residual ]
- final LayerNorm → linear head, **weights tied** to the token embedding
- causal masking via a registered lower-triangular buffer

Two configs:

| Config | layers | d_model | heads | ctx | params | purpose |
| --- | --- | --- | --- | --- | --- | --- |
| `small` | 6 | 384 | 6 | 256 | ~11.4M | the Mac / T4 run |
| `tiny` | 2 | 16 | 2 | 32 | ~0.04M | smoke tests only |

Context of 256 tokens is deliberate: at one token per ply, 256 tokens is ~128
plies, which covers a complete game. The model never has to reason about a
position whose origin has fallen out of its window.

### D4 — Data

Source: `lichess_db_standard_rated_2013-01.pgn.zst` (17.7 MB, ~121k games).
Chosen because it is the smallest official Lichess dump, so the ingest is
reproducible on a laptop in minutes rather than hours.

Filters: standard chess only, both players rated ≥ 1500, at least 10 plies,
game terminated normally. Games are converted PGN → UCI with python-chess,
which also validates them; anything that fails to parse is dropped.

Packing: `<bos> m1 m2 … mN <eos>` per game, concatenated into one flat stream,
chunked into windows of `ctx + 1`. Split 99/1 train/val **by game**, before
packing, so no game contributes to both sides.

### D5 — Training

Own loop. AdamW, cosine decay with linear warmup, gradient clipping at 1.0,
periodic checkpointing, resumable from a checkpoint (model + optimizer + step +
RNG state). Deterministic given a seed.

### D6 — Sampling

Temperature and top-k, implemented directly on the logits. Streaming: the
sampler is a generator that yields one decoded move at a time.

### D7 — Eval harness

Four metrics, each with a defined method:

1. **Validation loss** — mean cross-entropy over held-out windows.
2. **Legal-move rate** — sample N continuations from `<bos>`; at each step
   decode the token, ask python-chess whether that move is legal in the current
   position; count legal / total. On an illegal move, record it and **push a
   random legal move instead** so the game continues and later plies are still
   measured. (The alternative — stopping at the first illegal move — makes the
   rate a function of how early the model fails, double-counting metric 3.)
3. **Plies before first illegal move** — mean over the same N continuations.
4. **Baselines**, on the same protocol:
   - *uniform-random-token* — sample uniformly from the 1968 move tokens. This
     is the floor; it establishes what 0 knowledge looks like.
   - *bigram* — count-based P(next move | previous move) over the training
     corpus, backing off to unigram. This is the bar a transformer must clear
     to have justified its existence.
   - *random-legal* — pick uniformly among actual legal moves. Scores 100% by
     construction; included as the ceiling, and as a sanity check that the eval
     harness itself is not broken.

### D8 — CLI

`python -m nanochess {data,train,sample,eval}`.

The spec asked for three commands. `data` is a fourth, because ingest has to
live somewhere and hiding a 17 MB download inside `train` makes `train`
non-idempotent and untestable. Flagged here rather than silently added.

## Success criteria

- Full suite green, and fast — no test trains a real model.
- A real training run completes inside the budget and produces reported numbers.
- Legal-move rate materially beats the bigram baseline. A concrete target:
  **> 85% legal over the first 20 plies**. If the real number misses, the README
  reports the real number.
- Every tensor op in the model carries a shape assertion.
- `--seed` produces bit-identical results across runs on the same device.

## Non-goals

No web UI, no chat interface, no API server. No RLHF, DPO, or instruction
tuning. No distributed or multi-GPU training. No attempt to play strong chess —
Elo is not measured and is not a goal.
