# Plan 001 — nanochess implementation

Derived from `docs/specs/001-nanochess.md`. Strict TDD: each task writes a
failing test first, then the minimum code to pass it, then the full suite.
One commit per green cycle. No task exceeds ~100 lines of diff.

## Task list

| # | Task | Module | Test |
| --- | --- | --- | --- |
| 1 | Move vocabulary enumeration (1968 + 4 specials) | `tokenizer.py` | vocab size, no duplicates, contains known moves |
| 2 | `MoveTokenizer` encode/decode round-trip | `tokenizer.py` | round-trip on a real game, `<unk>` handling |
| 3 | Tokenizer save/load to disk | `tokenizer.py` | save → load → identical mapping |
| 4 | `CharTokenizer` behind the same protocol | `tokenizer.py` | round-trip, protocol conformance |
| 5 | Seed / device utilities | `utils.py` | same seed → same tensor; device resolution |
| 6 | PGN → UCI game extraction + filters | `data.py` | parses a fixture PGN, applies Elo/ply filters |
| 7 | Corpus download + streaming decompress | `data.py` | mocked stream, no network in tests |
| 8 | Packing into `ctx+1` windows, train/val split by game | `data.py` | window shapes, no game crosses the split |
| 9 | Batch sampler | `data.py` | shapes, x/y offset by one, determinism under seed |
| 10 | Causal self-attention | `model.py` | output shape, mask actually blocks the future |
| 11 | MLP block + transformer block | `model.py` | shapes, residual identity at zero-init |
| 12 | Full `NanoChessGPT` + weight tying | `model.py` | param count, tying is shared storage, loss decreases |
| 13 | Config presets `tiny` / `small` | `model.py` | param counts within expected range |
| 14 | LR schedule (warmup + cosine) | `train.py` | endpoints, monotonicity within each phase |
| 15 | Train step: forward, clip, step | `train.py` | loss decreases on a fixed batch (overfit test) |
| 16 | Checkpoint save/load | `train.py` | round-trip of model + optimizer + step |
| 17 | Resume produces identical trajectory | `train.py` | train 10, vs train 5 + resume 5 |
| 18 | Full training loop smoke test | `train.py` | `tiny` model, 10 steps, runs clean |
| 19 | Temperature + top-k logit transforms | `sample.py` | top-k masks correctly, temp → argmax at 0 |
| 20 | Streaming generator | `sample.py` | yields decoded moves one at a time |
| 21 | Legality checker over a token stream | `evaluate.py` | known-legal and known-illegal sequences |
| 22 | Legal-move-rate + plies-to-first-illegal | `evaluate.py` | hand-built stub model with known behaviour |
| 23 | Validation loss | `evaluate.py` | matches manual computation on a fixed batch |
| 24 | Random-legal and uniform-random baselines | `baselines.py` | random-legal scores exactly 1.0 |
| 25 | Bigram baseline | `baselines.py` | fits counts, backs off to unigram |
| 26 | CLI wiring for all four subcommands | `cli.py` | argument parsing, dispatch |
| 27 | CI workflow | `.github/workflows/ci.yml` | — |
| 28 | Real training run + README with actual numbers | `README.md` | — |

## Test discipline

- Any test touching a model uses the `tiny` config (2 layers, 16 dim, 32 ctx).
- No test hits the network; the download path is tested against a local stream.
- No test writes outside `tmp_path`.
- Target: the whole suite runs in seconds.

## Risks

- **Shape bugs are silent.** Mitigation: assertions on every tensor op in the
  model, not just at the boundaries.
- **The eval could be trivially satisfiable.** If the model learns "always play
  the most common opening move" it may score well early and badly later.
  Mitigation: report legal rate bucketed by ply, not just aggregate.
- **CPU-only CI cannot run the real thing.** Accepted: CI runs the suite, the
  reported numbers come from a real run on real hardware and the README states
  exactly which.
