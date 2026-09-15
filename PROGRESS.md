# Build progress — resume here

Snapshot of the Telecom Egypt Intelligent Assistant build. Everything below is
either **verified running** or **written but not yet executed** — the two are
kept strictly separate, because a lot of the code has never had a model behind it.

Environment: `smart_assistant/` venv on **Python 3.11.16**, 296 packages installed.

---

## Verified working (actually run, output observed)

| Thing | Evidence |
|---|---|
| Dependency install | 296 packages, `llama_cpp` and all imports resolve |
| Profile detection | `get_settings()` → `cpu-lite` on this machine |
| te.eg crawler | **600 pages** fetched (400 Arabic + 200 English), ~1.15 s/page |
| Boilerplate cleaning | 600 → 586 → **388 docs** after dedupe; FAQ tables preserved as Markdown |
| Chunking + embedding | **681 chunks** indexed in 39 s (17.3 chunks/s) |
| Hybrid retrieval | dense + BM25 + RRF, **24 ms median** query latency |
| **Retrieval eval** | **recall@5 = 81.6%**, MRR 0.759 over a 38-question bilingual golden set |
| Unit tests | **81 passing** (security, permissions, text processing, session isolation) |

### Measured retrieval, by query language

| Language | n | recall@5 | MRR |
|---|---|---|---|
| English | 10 | 90.0% | 0.833 |
| Mixed (code-switched) | 5 | 100.0% | 0.800 |
| MSA (`ar`) | 8 | 75.0% | 0.688 |
| Egyptian dialect (`arz`) | 15 | 73.3% | 0.733 |

Full report: `data/eval/results/rag.json`.

---

## Bugs found and fixed during the build

These cost real time and are worth not re-discovering:

1. **`llama-cpp-python` has no PyPI wheel** — it is sdist-only, so pip attempts a
   CMake build, which fails with no MSVC *and* trips Windows' 260-char path limit
   on the vendored llama.cpp tree. Fixed by pinning `<=0.3.19` and wiring the
   project's own wheel index via `[[tool.uv.index]]` with `explicit = true`.
   This is also why the venv is Python **3.11**, not the system 3.13.
2. **Arabic light stemming was missing** → BM25 treated "أشحن" / "شحن" / "الشحن"
   as unrelated terms, so naturally-phrased questions missed the page that
   answered them. Added `light_stem`. Then a second bug inside it: stripping
   single-letter prefixes ate the first root letter ("باقات"→"قات" vs
   "باقة"→"اقه"), so plural and singular stopped matching. Now multi-character
   article forms only. 11/11 conflation cases pass.
3. **Chunker silently dropped content** below `min_chars` — a document of short
   lines (a plan list, a code table) vanished entirely. Now merges backwards,
   but only while the result still fits the token budget (an unbounded merge
   produced one oversized mega-chunk).
4. **Corpus was 85% Arabic** because the seeds were Arabic-only; te.eg serves
   English under separate `/en/` URLs that link-following never reached. Added
   English seeds → English docs went 33 → 183, and English recall@5 is now 90%.
5. **One page was ~21% of the index** — the press-release listing is 183k chars.
   Excluded corporate-communications paths and capped any single document at 40k.
6. **fastembed defaulted to 1 ONNX thread** → indexing ran at 0.8 chunks/s.
   Pinned to physical cores: **17–21 chunks/s**, a 12-minute build became 39 s.
7. `fastembed` does not ship `multilingual-e5-small`; using
   `paraphrase-multilingual-MiniLM-L12-v2` (384-d, 0.22 GB) instead — chosen on
   footprint, see the note in `config.py`.

---

## Completed since the first snapshot

- `monitoring/metrics.py` + `/metrics` and `/metrics/{session_id}` endpoints — per-session
  latency, WER and quality counters, wired into every pipeline exit path
- Tests for frustration/handoff, intent-parsing robustness and citation resolution
  (**126 passing**, up from 81)
- `scripts/`: `fetch_models.py`, `eval_asr.py`, `bench_models.py`, `run_local.ps1`,
  `run_local.sh`
- `Dockerfile`, `docker-compose.yml`, `.dockerignore` — three services, one image
- `notebooks/te_assistant_walkthrough.ipynb` — **A.5 deliverable** and Colab launcher
- `docs/presentation-outline.md` — **A.5 deliverable**, 20 slides with speaker notes
- `RUNBOOK.md` — every command, four install paths, troubleshooting, demo script
- `uv.lock` (186 packages), `.gitignore`, **git repository initialised and committed**
- `ruff` clean across `src`, `tests`, `scripts`

### One more bug found

`_clean_slots` digit-stripped the `msisdn` slot, which reduced `<PII_PHONE_1>` to `1` —
destroying the placeholder before the backend could unmask it, so account resolution would
silently fail on any masked phone number. Caught by a test written for exactly that path.

---

## Written but NOT yet executed

**No language model, ASR model, or TTS voice has ever been loaded.** The code is
written and imports cleanly, but these paths are unverified:

- `llm/` — client, prompts, intent detection, grounded generation with citations
- `speech/` — ASR, hallucination filter, TTS, the speech service
- `documents/` — PDF/DOCX/image parsing and OCR
- `pipeline.py` — the full turn orchestrator (guardrails → PII → intent →
  permissions → answer)
- `api.py` — core FastAPI service
- `ui/gradio_app.py` — the interface

The blocker is a **2 GB GGUF download** (`Qwen/Qwen2.5-3B-Instruct-GGUF`,
`qwen2.5-3b-instruct-q4_k_m.gguf` — filename verified to exist in the repo).
Disk at last check: **7.78 GB free**.

---

## Next steps, in order

1. **Download the models** (~2.7 GB total, all resumable):
   ```
   python -c "from huggingface_hub import hf_hub_download as d; d('Qwen/Qwen2.5-3B-Instruct-GGUF','qwen2.5-3b-instruct-q4_k_m.gguf')"
   ```
   Whisper `small` (~0.5 GB) and the Piper voices (~0.12 GB) fetch themselves on
   first use.
2. **End-to-end smoke test**: start `te-core`, POST `/chat`, confirm a grounded
   Arabic answer comes back with resolvable citations. This is the first real
   test of `pipeline.py`, `llm/generate.py` and the citation resolver.
3. **Measure and record CPU latency** per stage — the B.10 budget is currently
   an estimate (~20–25 s/answer) and needs to become a measured number.
4. **Speech**: warm the speech service, transcribe a dialect clip, synthesise a
   reply. Verify the hallucination filter fires on a silent clip.
5. **Then**: the seven remaining deliverable pieces listed below.

---

## Not yet written

| Item | Referenced by |
|---|---|
| `data/eval/audio/manifest.jsonl` + clips | needed before `eval_asr.py` produces numbers |
| Tests for document parsing / OCR | B.7 — needs sample PDFs |
| Redis-backed `SessionManager` | horizontal scaling (see gaps below) |
| `git remote` + push | Colab workflow — needs your GitHub URL |

---

## Known gaps to be honest about in the presentation

- **Egyptian-dialect retrieval is the weakest language at 73.3%**, and the seven
  misses are all pages that *are* in the corpus but rank below 5 — an embedder
  limitation, not a coverage gap. BGE-M3 on the GPU profile should close much of
  it; that comparison is exactly what `bench_models.py` is for.
- Reranking is **off** on `cpu-lite` (too slow on 4 cores) — a documented cut.
- Guardrails are pattern-based and stop scripted attempts, not a determined
  adversary. The architecture does not rely on them alone.
- **The session store is in-process.** `core` is described as stateless and
  horizontally scalable, but a second replica would break session affinity today.
  The interface is narrow enough that Redis is a one-file swap — but as it stands,
  horizontal scaling is a design property, not a demonstrated one. Say it that way.
- **Docker is unbuilt.** Docker is not installed here and Colab cannot run nested
  containers, so the first `docker compose build` is the real test.
- **Colab CPU ≠ this laptop.** Colab's CPU runtime is ~2 vCPU. If `cpu-lite` latency is
  measured there, state the hardware rather than presenting it as the on-premises figure.

---

## Resuming on Colab

1. `git remote add origin <your-repo>` && `git push -u origin main`
2. Open `notebooks/te_assistant_walkthrough.ipynb` in Colab, set `REPO_URL`, pick a T4
3. Run all — it clones, installs, **re-indexes** (required: the committed index is 384-d,
   the GPU profile is 1024-d), fetches models, and launches with a public URL

Full detail in [RUNBOOK.md §2](RUNBOOK.md#2-google-colab).
