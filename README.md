# Telecom Egypt — Intelligent Assistant

A bilingual (Arabic / English / Egyptian dialect) voice-and-text assistant that answers
customer questions from te.eg content and from documents the user uploads, with source
citations on every answer. **The core runs fully on-premises** — no external API is
required at inference time.

> Built as a proof of concept for the Telecom Egypt AI case study. External APIs appear
> only in the benchmark notebook, for comparison, never in the serving path.

**📖 [RUNBOOK.md](RUNBOOK.md) is the complete command reference** — local, Colab, Docker,
air-gapped, troubleshooting and the demo script. This README is the overview.

| I want to… | Go to |
|---|---|
| **Test it right now on Colab** | **[COLAB.md](COLAB.md)** — step by step, ~15 min |
| Run it locally | [Local install](#local-install) · [RUNBOOK §1](RUNBOOK.md#1-local) |
| Deploy to a server | [RUNBOOK §3 — Docker](RUNBOOK.md#3-docker) |
| Install with no internet | [RUNBOOK §4 — air-gapped](RUNBOOK.md#4-air-gapped-install) |
| Understand the pipeline | [the notebook](notebooks/te_assistant_walkthrough.ipynb) |
| Present it | [**the deck**](docs/TelecomEgypt_Assistant.pptx) · [outline + notes](docs/presentation-outline.md) |
| **Publish this repo** | **[DATA_NOTICE.md](DATA_NOTICE.md)** — read before going public |
| Fix something | [RUNBOOK §8 — troubleshooting](RUNBOOK.md#8-troubleshooting) |

---

## Measured results

`cpu-lite` profile, 38-question bilingual golden set, 681 indexed chunks:

| | |
|---|---|
| **recall@5** | **81.6%** |
| **MRR** | 0.759 |
| **Median retrieval latency** | 18–24 ms |
| **Tests** | 126 passing |

| Query language | n | recall@5 |
|---|---|---|
| English | 10 | 90.0% |
| Code-switched (AR/EN) | 5 | 100.0% |
| MSA | 8 | 75.0% |
| Egyptian dialect | 15 | 73.3% |

Reported split by language deliberately — an overall figure that hid "English retrieves
nothing" would be misleading. Reproduce with `python scripts/eval_rag.py --k 5`.

---

## What it does

- **Voice or text in.** Speech is transcribed locally; the input mode is known from the
  channel the request arrives on.
- **Voice and text out.** Spoken answers are always accompanied by the text — never
  voice-only. Text questions do not trigger speech synthesis.
- **Grounded answers with citations.** Every answer cites the te.eg pages or uploaded
  documents it came from, and the system refuses rather than guessing when retrieval
  returns nothing relevant.
- **Documents.** PDF, DOCX, TXT and images are parsed (with OCR where needed), embedded,
  and scoped to the uploading session only.
- **Arabic, English, and Egyptian dialect**, including code-switched input.
- **Actions without giving the model database access.** The model detects intent; the
  backend checks permissions and executes.
- **Frustration detection** with a defined handoff path to a human agent.

---

## Requirements

| | |
|---|---|
| Python | **3.11+**; the CPU profile needs exactly **3.11 on Windows** (see below) |
| Disk | ~5 GB for the CPU profile (environment + models) |
| RAM | 8 GB minimum, 16 GB comfortable |
| GPU | Optional. The CPU profile is the supported baseline. |
| ffmpeg | Required for audio decoding — must be on `PATH` |

### Why 3.11 on Windows

Only the **`cpu` extra** is version-sensitive, because of `llama-cpp-python`. It publishes
**source distributions only** to PyPI, so a plain install triggers a CMake/C++ build. That
needs a compiler, and on Windows it also trips the 260-character path limit while unpacking
the vendored `llama.cpp` tree. Prebuilt wheels live on the project's own index, and the
newest Windows one targets CPython 3.11 — `pyproject.toml` wires that index up for this
one package.

The **core** and the **`gpu` extra** have no such constraint and run on 3.11 through 3.13.
That separation matters: pinning it project-wide previously made the package uninstallable
on Colab, which runs 3.13.

### Which extras to install

| Target | Command |
|---|---|
| Local / on-premises CPU | `uv pip install -e ".[cpu,dev]"` |
| Colab / any CUDA GPU | `pip install -e ".[gpu]"` — **not** `[cpu]`; the GPU profile never loads llama.cpp |
| Core only (retrieval, no local LLM) | `pip install -e .` |

---

## Quick start

```bash
pip install uv && uv python install 3.11
uv venv smart_assistant --python 3.11
.\smart_assistant\Scripts\activate          # or: source smart_assistant/bin/activate
uv pip install -e ".[cpu,dev]"
python scripts/fetch_models.py              # ~2.7 GB, once

.\scripts\run_local.ps1                     # or: ./scripts/run_local.sh
```

Then open <http://127.0.0.1:7860>. The knowledge base ships with the repository, so there
is no crawl step.

**Docker:**

```bash
docker compose build
docker compose run --rm core python scripts/fetch_models.py
docker compose up -d
```

> The Docker setup has **not been built or run** — Docker is unavailable on the development
> machine and Colab cannot run nested containers. It is a deployment artifact, not a
> verified one.

---

## Local install

```bash
git clone <this-repo> te-assistant
cd te-assistant

# uv provisions the right interpreter itself — no system Python 3.11 needed
pip install uv
uv python install 3.11
uv venv smart_assistant --python 3.11

# Windows
.\smart_assistant\Scripts\activate
# macOS / Linux
source smart_assistant/bin/activate

uv pip install -e ".[cpu,dev]"
```

`uv.lock` is committed, so installs are byte-for-byte reproducible.

### Offline / air-gapped install

The on-premises target usually has no internet. Build a wheel bundle on a connected
machine and carry it across:

```bash
# connected machine
uv pip download -e ".[cpu,dev]" -d wheelhouse/
python scripts/fetch_models.py --dest models/     # pulls the GGUF, ONNX and voice files

# air-gapped machine
uv pip install --no-index --find-links wheelhouse/ -e ".[cpu,dev]"
export TE_MODELS_DIR=/path/to/models
```

Nothing in the serving path reaches the network once the models are on disk.

---

## Running

Three processes. Start them in this order:

```bash
# 1. one-time: build the te.eg knowledge base (skip if data/chroma/ is populated)
te-ingest

# 2. speech service  (ASR + TTS)      :8001
te-speech

# 3. core service    (RAG, security)  :8000
te-core

# 4. UI                               :7860
te-ui
```

Or on Windows, `scripts/run_local.ps1` starts all three.

The knowledge base is **committed to the repository**, so `te-ingest` is only needed if
you want to rebuild or re-crawl. Re-crawling takes roughly 20-30 minutes.

If the speech service is not running, the UI hides the microphone and text chat continues
to work. That degradation is deliberate and is covered by a test.

---

## Profiles

One codebase, two runtime profiles, selected automatically by device detection:

| | `cpu-lite` | `gpu-colab` |
|---|---|---|
| Selected when | no CUDA GPU, or < 12 GB VRAM | CUDA GPU with ≥ 12 GB VRAM |
| ASR | faster-whisper `small`, int8 | faster-whisper `large-v3`, fp16 |
| Embeddings | `multilingual-e5-small` (ONNX) | `BAAI/bge-m3` |
| Reranker | off — RRF ordering is final | `BAAI/bge-reranker-v2-m3` |
| LLM | `Qwen2.5-3B-Instruct` Q4\_K\_M | `Qwen3-8B-AWQ` |
| TTS | Piper | Piper |

Override with `TE_PROFILE=cpu-lite` or `TE_PROFILE=gpu-colab`. Forcing `cpu-lite` on a GPU
machine is how the on-premises latency figures are measured rather than estimated.

`cpu-lite` is deliberately **torch-free** — CTranslate2, ONNX Runtime and llama.cpp only.
That keeps the install near 1.5 GB instead of 4.5 GB, which is what makes it a credible
on-premises artifact. Installing `.[gpu]` pulls PyTorch and is only needed for the GPU
profile.

### Every core model is commercially licensed

| Model | License |
|---|---|
| Qwen2.5-3B-Instruct / Qwen3-8B-AWQ | Apache-2.0 |
| multilingual-e5-small | MIT |
| BAAI/bge-m3, bge-reranker-v2-m3 | MIT / Apache-2.0 |
| faster-whisper (Systran) | MIT |
| Piper voices | MIT |

`jina-embeddings-v3`, `jina-reranker-v2` and `f5-tts-egyptian-arabic` are **CC-BY-NC-4.0**
and are therefore excluded from the serving path. They appear only in
`scripts/bench_models.py`, as comparison baselines.

---

## Running on Google Colab

```python
!git clone <this-repo> && cd te-assistant && pip install -e ".[gpu]"
!python -m te_assistant.ui.gradio_app --share
```

The knowledge base ships with the repository, so there is no crawl or re-embed step. The
GPU profile is detected automatically. Note that a T4 is Turing: fp16 only, no bf16 and no
flash-attention, and the full model set does not fit resident in 16 GB — OCR and TTS are
loaded on demand and released.

---

## Configuration

Everything is environment-overridable; defaults live in `src/te_assistant/config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `TE_PROFILE` | auto | `cpu-lite` or `gpu-colab` |
| `TE_DATA_DIR` | `./data` | corpus, index, sessions, traces |
| `TE_CORE_URL` | `http://127.0.0.1:8000` | where the UI finds core |
| `TE_SPEECH_URL` | `http://127.0.0.1:8001` | where core finds speech |

---

## Tests

```bash
uv run pytest
```

The suite covers each service, and asserts the **negative** cases specifically: prompt
injection is refused, PII never reaches the model, an unauthorised action is refused even
when intent detection succeeds, and one session cannot retrieve another session's
documents.

## Evaluation

```bash
uv run python scripts/eval_rag.py     # recall@k, citation correctness, groundedness
uv run python scripts/eval_asr.py     # WER on noisy Egyptian-dialect clips
uv run python scripts/bench_models.py # model comparison, incl. external-API baselines
```

Results tables are written to `data/eval/results/`.

---

## Layout

```
src/te_assistant/
  config.py        profile + device detection, model slots
  schemas.py       the contract between the three services
  ingest/          te.eg crawl, boilerplate removal, chunking, index build
  retrieval/       vector store, hybrid BM25+dense, reranking, Arabic normalisation
  documents/       PDF/DOCX/TXT/image parsing, OCR, per-session ingestion
  security/        guardrails, PII masking, permissions
  actions/         intent registry and the SQLite the backend (not the model) owns
  llm/             model client, prompts, intent detection, grounded generation
  speech/          VAD, ASR, hallucination filtering, TTS
  session/         session lifecycle and eviction
  frustration/     frustration scoring and human handoff
  insights/        per-turn structured records and the session insights view
  ui/              Gradio application
```

## Known limitations

- Guardrails are pattern-based. They stop common scripted injection attempts, not a
  determined adversary with a paraphrase budget. The architecture does not depend on
  them alone — the model has no database access regardless of what it is persuaded to say.
- There is no user authentication. Sessions are anonymous and unguessable; per-document
  ownership is future work.
- The te.eg snapshot is point-in-time. Prices and offers drift; re-run `te-ingest`.
- On CPU, expect roughly 20-25 s per answer and 35-45 s voice-to-voice on a 4-core laptop.
  Measured figures for both profiles are in `data/eval/results/`.
