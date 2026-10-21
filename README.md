# 基于大模型后训练的指令微调自进化系统 RSI

**RSI (Recursive Self-Improvement) for Instruction Tuning**: an automated optimization system for post-training fine-tuning.
This pilot covers instruction tuning, running on a single RTX 4090 with Qwen3-1.7B-Base.

**Private repository:** https://github.com/notquitezero-ai/rsi-instruct-evolve

**Static demo (no GPU needed):** the exported site is in `docs/`. Preview it with
`python -m http.server 8000 --directory docs`, then open http://127.0.0.1:8000/.
Uploading the repository does not itself enable a publicly accessible GitHub Pages site.
Demo deck: [`deliverables/RSI_演示.pptx`](deliverables/RSI_演示.pptx) · System iteration log: [`docs/系统迭代记录.md`](docs/系统迭代记录.md)

Each generation, the current best model (the "champion") **writes its own prompts, answers them, and has a program verify the answers; the answers that pass become the next generation's training data**. A meta-optimizer proposes candidate training recipes. Candidates are trained and then compared with the champion on a freshly drawn gate set, and a statistical gate decides whether one is promoted. The whole loop needs no human labelling.

```
Diagnose weaknesses → write prompts (curriculum-weighted) → sample k answers → programmatic verification → add to self-generated data pool
        ↑                                                                            ↓
 Reflect + tune the data engine ← statistical gate (promote / roll back) ← re-test with the champion on a fresh gate set ← train N candidate recipes (LoRA)
                                                                                     ↑
                                                         Meta-optimizer (Thompson sampling over mutation operators + failure-directed repair)
```

## Results (pilot run: 10 generations, 3 promotions, 29 candidate models trained and evaluated, ~11 GPU hours)

**In-project test set** (200 constrained instructions + 150 GSM8K; **never used for any selection or promotion decision**):

| Metric | Original Qwen3-1.7B-Base | Gen 9 | Gen 10 (current champion) |
|---|---|---|---|
| Strict instruction-following accuracy | 26.5% | 48.0% | 46.0% |
| Robust view (excluding verifier-gaming answers under 20 words) | 21.5% | 41.5% | 40.0% |
| Per-constraint accuracy | 45.2% | 64.5% | 63.5% |
| GSM8K (guard metric) | 72.7% | 82.7% | 80.7% |
| Degeneration rate (fails to stop / repeats; lower is better) | 23.1% | 6.9% | 9.1% |

**Third-party check: official IFEval** (google/IFEval, all 541 prompts, scored with lm-eval's official IFEval code):

| Official IFEval | Base | Gen 6 | Gen 9 | Gen 10 |
|---|---|---|---|---|
| Prompt-level strict (main metric) | 24.4% | 35.7% | 42.0% | **42.3%** (+17.9, 90% CI [+13.9, +22.0]) |
| Instruction-level strict | 37.4% | 48.1% | 53.0% | 54.1% |
| Instruction types never seen in training (9 types, instruction-level) | 35.6% | 44.1% | 47.4% | 50.4% |
| Prompt-level strict excluding passes under 20 words | 19.8% | 28.3% | 33.3% | 35.1% |

Gen 9 and gen 10 are at the same level: gen 9 is slightly ahead on the in-project test set, gen 10 slightly ahead on IFEval, and both gaps are within noise. Gen 10 is the champion the system chose by its own rules; the headline was **not** switched to whichever model scored best on test, because that would itself be selecting on the test set. IFEval generation uses ChatML + greedy decoding + at most 1280 tokens, the same as this project's own evals. That differs from the common leaderboard prompt format, so absolute numbers are not directly comparable to leaderboards.

## The system itself evolves too (v1 → v7; details in [docs/系统迭代记录.md](docs/系统迭代记录.md))

| Version | Evidence | Change |
|---|---|---|
| v1 | After fine-tuning the model never stopped; training was 20× slower than expected | Switched to `<\|endoftext\|>` as the stop token; capped VRAM; chunked loss over response tokens only |
| v2 | Candidates were clearly better on the composite score but blocked by math regression | Select only among gate-passing candidates; failure-directed repair |
| v3 | The candidate closest to passing was not chosen as the repair parent | Parent = smallest guard violation; data-source ablation operators |
| v4 | Every candidate regressed on math | Self-distilled seed data (the system's own experiments later refuted this hypothesis); new operators are tried once first |
| v5 | Median math answer length fell 217 → 117 tokens | **Root cause: STaR kept the "shortest correct solution", rewarding lucky guesses.** Now validates the reasoning chain and takes the median-length solution, with retroactive cleanup |
| v6 | A champion answering only "P.S." passed 3 format checks | Substance filter, plus a robust metric |
| v7 | The champion's dev math was 85.3% but only 76.0% on fresh items (winner's curse) | A fresh gate set is drawn each generation, and the champion is re-tested side by side with the candidates |

The promotion criteria were never loosened during the run.

## Code structure

| Module | File | Role |
|---|---|---|
| Verifiable constraint library | `rsi/constraints.py` | 18 IFEval-style constraints: render the prompt sentence / deterministic check / rewrite |
| Data and evaluation sets | `rsi/data.py` | alpaca-cleaned / gsm8k split deterministically into train, dev, gate, and test |
| Self-data engine | `rsi/synth.py` | Rejection-sampling fine-tuning (ReST-EM), STaR with reasoning-chain validation, self-distilled seed data, curriculum weighting, substance filter |
| Trainer | `rsi/train.py` | LoRA SFT; loss on response tokens only, chunked LM-head computation, token-budget batching |
| Evaluator | `rsi/evaluate.py` | Strict / per-constraint / robust metrics, GSM8K, degeneration rate, paired bootstrap |
| Meta-optimizer | `rsi/optimizer.py` | Mutation operators, Thompson sampling, failure-directed repair, gate, data-engine feedback control |
| Orchestrator | `rsi/loop.py` | Resumable generation loop (state in `runs/<run>/state.json`) |
| Official IFEval | `rsi/ifeval_official.py` | Third-party check plus seen/unseen-type and short-answer breakdown |
| Interactive demo | `app/demo.py` | Gradio: overview / diagnostics / sample comparison / **live chat** / evolution console |
| Static site | `scripts/build_site.py` → `docs/` | The GitHub Pages online demo |
| Demo deck | `scripts/make_ppt.py` → `deliverables/` | Generated automatically from run data |

## Reproduce

Python 3.11, CUDA GPU (24 GB recommended):

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft datasets accelerate gradio python-pptx matplotlib pandas pyyaml pytest "lm-eval[ifeval]"
python -m rsi.prepare                      # download Qwen3-1.7B-Base into models/ and build data/prepared/
python -m pytest -q tests
python -m rsi.loop --config configs/smoke.yaml     # end-to-end self-check in a few minutes
python -m rsi.loop --config configs/pilot.yaml     # self-evolution (resumable; --generations N adds more)
python -m rsi.ifeval_official                      # official IFEval: base vs current champion
python app/demo.py --run pilot                     # full demo at http://127.0.0.1:7860
python scripts/build_site.py --run pilot           # rebuild the static site in docs/
python scripts/make_ppt.py --run pilot             # rebuild the demo deck
```

On Windows you can first `. .\scripts\env.ps1` to point pip / HuggingFace / torch / temp caches into the project directory (avoiding drive C).
Model weights, LoRA adapters, raw data, and full run artifacts are not in the repo (see `.gitignore`); the commands above regenerate them.

## Evaluation and honesty constraints

- **test** is evaluated only on the baseline and on promoted champions, and never feeds back into any decision. **gate** is drawn fresh each generation and used for promotion decisions (v7). **dev** is used only for curves and diagnostics.
- Promotion requires all of: composite score Δ > 0, paired bootstrap P(better) ≥ 0.8, math drop ≤ 3 points, degeneration rise ≤ 3 points.
- The verifiers check format and constraints only, not content quality, and can be gamed. The robust metric and the official IFEval check are reported alongside the main numbers.

## Data and model licenses

Base model Qwen3-1.7B-Base (Apache-2.0); data yahma/alpaca-cleaned (CC BY 4.0), openai/gsm8k (MIT), google/IFEval (Apache-2.0). The sample outputs in the static site come from this project's actual evaluation runs.
