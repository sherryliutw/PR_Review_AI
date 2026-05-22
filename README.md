# PR Review AI

Fine-tune an open-source LLM on 355K real human code reviews from GitHub, then ship it as a GitHub App that auto-reviews Pull Requests. The full system — model training, inference API, webhook handler, and Kubernetes deployment — is built end-to-end.

> **Status:** Phase 1 (training), Phase 2 (API), and Phase 3 (K8s) are complete and verified end-to-end. Currently iterating on model quality (v3 training in progress).

---

## Why this project

Most "LLM portfolio" projects stop at a notebook. This one goes from raw HuggingFace dataset → QLoRA-fine-tuned adapter → FastAPI webhook service → Docker image → Helm chart → GitHub Actions CI/CD. It also documents what *didn't* work, because the iteration story is the interesting part.

---

## Architecture

Three layers, each independently deployable:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — Training (Colab Pro, T4 GPU)                     │
│  HF dataset → filter → QLoRA fine-tune → 65 MB adapter      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 — Backend (FastAPI + PostgreSQL)                   │
│  GitHub webhook → HMAC verify → fetch PR diff →             │
│  model inference → post review comments → save history      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — Deployment (Docker + Helm + GitHub Actions)      │
│  Multi-stage CUDA image · Bitnami PostgreSQL subchart ·     │
│  GPU resource requests · CI builds & pushes to GHCR         │
└─────────────────────────────────────────────────────────────┘
```

---

## Results (v1 → v2 → v3)

| Version | Training config | Eval score | Key finding |
|---|---|---|---|
| **v1** | 1 epoch, 10K samples, vanilla HF + PEFT + TRL | 1/3 on hand-picked tests | Identifies some patterns; misses security issues |
| **v2** | 2 epochs, 15K samples, **Unsloth** (~2× speedup) | ~3/10 on expanded 10-case set, unstable | Validation loss flat after epoch 1 — more epochs ≠ better |
| **v3** (in progress) | 1 epoch, 15K positive-only samples | TBD | Hypothesis: removing `is_negative=True` (25% of data) will reduce "no issues found" bias |

The v2 evaluation surfaced two important findings:
1. **The model was systematically biased toward saying "no issues found"** because 25% of training data was negative examples. v3 tests whether dropping these helps.
2. **Eval cases like SQL injection (0.37% of data) and N+1 queries (0.01%) were unfair** — the model was never taught these patterns. I split the eval into *fair* (well-represented) and *stretch* (under-represented) buckets so the score actually means something.

---

## Tech stack

| Layer | Tools |
|---|---|
| **Fine-tuning** | CodeLlama-7B-Instruct, QLoRA (4-bit), Unsloth, bitsandbytes, PEFT |
| **Training data** | [`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview) (355K rows, filtered to Python + `quality_score ≥ 0.5`) |
| **Inference** | Transformers + PEFT (direct), CUDA 12.1 |
| **API** | FastAPI, httpx, PyJWT (GitHub App auth), pydantic-settings |
| **Database** | PostgreSQL, async SQLAlchemy + asyncpg |
| **Infra** | Docker (multi-stage), Helm (Bitnami PostgreSQL subchart), Kubernetes (`nvidia.com/gpu`) |
| **CI/CD** | GitHub Actions → GHCR → Helm deploy |
| **Dev tunnel** | smee.io (forwards GitHub webhooks to Colab) |

---

## Key design decisions

A few non-obvious calls and the reasoning behind them:

- **QLoRA over full fine-tuning.** Trainable params are ~0.06% of total. Fits on a free-tier T4 (16 GB VRAM). The output adapter is 65 MB instead of 14 GB.
- **`before_code` + `after_code` over raw diffs.** Gives the model full context, mimicking how a human reviewer reads code. Diffs alone are too compressed.
- **Unsloth for v2.** Custom Triton kernels for LoRA forward/backward give ~2× training speedup and ~40–50% less VRAM with identical accuracy. Adapters stay compatible with vanilla `peft.PeftModel.from_pretrained`, so the API server doesn't need Unsloth as a runtime dependency.
- **Direct Transformers + PEFT inference (not vLLM).** For v1, simpler beats faster. vLLM/TGI migration is on the roadmap once latency under real traffic becomes the bottleneck.
- **Background-task webhook handling.** GitHub kills webhooks at 10s. The handler returns 202 immediately and runs the review pipeline async to stay within the budget.
- **Dry-run mode (`SKIP_MODEL_LOAD=true`).** Lets the full API stack — webhooks, DB, GitHub calls — run on CPU/macOS for local development. Inference returns a placeholder. Made the Phase 3 Helm dry-run possible without a GPU cluster.

---

## License

MIT — see [LICENSE](LICENSE).
