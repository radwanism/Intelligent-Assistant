# Presentation outline — Telecom Egypt Intelligent Assistant

**Target: 18–20 slides, ~25 minutes, leaving 10 for questions.**

The evaluation notes say candidates are assessed on *technical approach, problem-solving,
system design thinking, and on-premises deployment considerations*. So the through-line is
**decisions and the evidence behind them**, not a feature tour. Anything with a measured
number attached goes in; anything that is only a diagram gets cut first.

Speaker notes in _italics_.

---

## 1 — Title

**Telecom Egypt Intelligent Assistant**
Bilingual voice + text RAG assistant · runs fully on-premises · Arabic, English, Egyptian dialect

---

## 2 — The problem, stated as constraints

- Thousands of daily inquiries in **Arabic, English and Egyptian dialect**
- Voice *and* text, with **noisy real-world audio**
- Answers must be **grounded in te.eg** and **cited** — a telecom cannot quote a wrong price
- Customers upload documents and ask about them
- **No external API in the serving path** — customer audio and documents stay on-premises

_"That last constraint is the one that shapes everything. It rules out the easy answer and
forces every model choice to be defended on footprint and latency."_

---

## 3 — What it does (30-second demo clip or screenshot)

Voice in → transcript shown → grounded answer with sources → spoken reply.
**Text is always displayed. Never voice-only.**

---

## 4 — Architecture: three services, and why only three

```
┌──────────┐      ┌──────────────────────────────────────┐      ┌──────────┐
│    UI    │─────▶│                CORE                  │─────▶│  SQLite  │
│  Gradio  │      │  guardrails · PII · retrieval ·      │      │ accounts │
│  :7860   │◀─────│  intent · permissions · sessions     │◀─────│ tickets  │
└────┬─────┘      └──────────────┬───────────────────────┘      └──────────┘
     │                           │
     │                    ┌──────▼───────┐
     │                    │   ChromaDB   │  te.eg KB + per-session uploads
     │                    └──────────────┘
     │            ┌──────────────┐
     └───────────▶│    SPEECH    │  ASR (faster-whisper) + TTS (Piper)
                  │    :8001     │
                  └──────────────┘
```

**Split only where failure isolation earns it.** Speech holds the heaviest models, has the
slowest cold start, and is the likeliest to be OOM-killed. Everything else shares the
session store and vector index, so splitting it would add network hops and buy nothing.

_"The question I asked for each boundary was: what breaks if this dies? For speech the
answer is 'voice input' — not 'the product'. That's a service. For retrieval and intent the
answer is 'everything', so they stay together."_

---

## 5 — The degradation contract

| Failure | Result |
|---|---|
| Speech service down | Mic hidden, **text chat unaffected** |
| LLM unavailable | Retrieval still returns cited passages |
| Reranker unavailable | Falls back to RRF ordering |
| TTS fails | Text answer still delivered |

**Demonstrable:** `docker compose stop speech` — the assistant keeps working.

_"A degradation story that's never exercised is a guess. Each row is a test."_

---

## 6 — Data flow: one voice turn

```
mic → VAD → ASR → hallucination filter → confidence gate
    → guardrails → PII mask → session resolve
    → hybrid retrieve (BM25 + dense → RRF) → rerank
    → intent ──action?──▶ permission check ──▶ backend executes
              └─else────▶ grounded generation + citations
    → output guardrail → unmask for display
    → frustration score → structured turn record
    → text + citations shown, TTS added
```

---

## 7 — Data quality: where retrieval quality is actually won

te.eg is a Liferay portal: **the same ~2–3k character menu on every page**, about a third
of a typical page.

Leave it in and every chunk shares most of its tokens with every other chunk — dense
similarity rises between unrelated pages and stops discriminating. BM25 degrades too,
because the menu vocabulary (*موبايل*, *إنترنت*) is exactly what users search for.

**Fix: corpus-level, not per-page.** Any line on >35% of pages is template furniture,
whatever the markup says.

Also: **tables preserved as Markdown**. `Nitro 200 | 300 EGP | 200 GB` flattened is a
number with no label, and the model quotes the wrong price.

| | |
|---|---|
| Pages crawled | 600 |
| After cleaning + dedupe | **388 documents** |
| Indexed chunks | **681** |
| One page was 21% of the index | press-release listing, 183k chars — excluded |

---

## 8 — Bilingual corpus: a bug worth showing

The first crawl was **85% Arabic**. te.eg serves English under separate `/en/` URLs that
Arabic-seeded link-following never reaches.

**English documents: 33 → 183. English recall@5: poor → 90%.**

_"The case study asks for bilingual output. That has to mean a bilingual corpus, not an
Arabic corpus queried in English and left to the embedder to bridge."_

---

## 9 — Arabic normalisation and light stemming

BM25 is lexical: «إنترنت» and «انترنت» are different terms to it.

Arabic is agglutinative — untreated, **«أشحن» / «شحن» / «الشحن» are three unrelated
terms**, so a naturally-phrased question misses the page that answers it.

**And a bug worth admitting:** the first stemmer stripped single-letter prefixes too, which
ate the first root letter — «باقات» → «قات» but «باقة» → «اقه». The plural and singular of
the word this corpus is most asked about stopped matching. Now multi-character article
forms only. 11/11 conflation cases pass.

_"Showing the fix is stronger than showing the feature. It's evidence of testing, not luck."_

---

## 10 — Hybrid retrieval: two failure modes, two retrievers

| | Fails at |
|---|---|
| **Dense alone** | Exact identifiers — "Nitro 200" retrieves "Nitro 100" |
| **BM25 alone** | Cross-lingual — «عايز أعرف أسعار النت» has zero lexical overlap with an English tariff page |

Fused with **Reciprocal Rank Fusion**, not a weighted sum: the two scores aren't on
comparable scales and per-query normalisation is fragile. RRF only needs ranks.

Plus: per-URL diversification, so one long FAQ page can't fill the whole context window.

---

## 11 — Measured retrieval results

38-question bilingual golden set, `cpu-lite` profile:

| | |
|---|---|
| **recall@5** | **81.6%** |
| **MRR** | **0.759** |
| **Median latency** | **24 ms** |

| Query language | n | recall@5 |
|---|---|---|
| English | 10 | 90.0% |
| Code-switched | 5 | 100.0% |
| MSA | 8 | 75.0% |
| **Egyptian dialect** | 15 | **73.3%** |

**Reported split by language on purpose.** An overall number that hides "English retrieves
nothing" would be misleading.

_"Dialect is the weakest, and I'll come back to why."_

---

## 12 — ASR: built for noisy dialect, not a Whisper wrapper

1. **VAD before transcription** — Whisper's failure on silence isn't silence, it's a
   confident sentence learned from subtitle corpora
2. **Egyptian-dialect `initial_prompt`** — Whisper drifts to MSA because MSA dominates its
   Arabic training data
3. **Hallucination filters** — known artifacts («ترجمة نانسي قنقر», "Thank you for
   watching") plus a repetition-loop detector
4. **Confidence gate** — ask the user to repeat rather than answer a misheard question

**Never `task="translate"`.** Pivoting dialect through English discards exactly what the
case study is testing.

---

## 13 — Security: the architectural claim

```
1. input guardrails   →  BEFORE the model sees anything
2. PII masking        →  the model never sees raw sensitive values
3. intent detection   →  the model's ONLY job
4. permission check   →  the backend decides
5. execution          →  the backend acts, never the model
```

**Steps 1 and 2 cannot be reordered.** Masking first feeds an injection string to the
masker; guarding after the model defeats the point.

> **The model's output carries no authority.** An intent can arrive perfectly formed,
> maximally confident, naming a real account — and it still does not execute unless that
> session holds the grant.

_Live demo: a correctly-detected `check_bill_balance` for someone else's account, refused._

---

## 14 — Security, evidenced

- **2 real action intents** over SQLite: `check_bill_balance` (read), `create_support_ticket` (write)
- PII by regex with a **Luhn check** — so an order reference isn't masked as a card, because
  over-masking removes information the model needs
- **Every refusal audited.** An audit log that only records successes can't answer "did
  anyone try?"
- **Tests assert the negative cases** — injection blocked, PII never reaches the model,
  unauthorised action refused, cross-session retrieval impossible

**126 tests passing.**

**Honest limit:** guardrails are pattern-based. They stop scripted attempts, not a
determined adversary with a paraphrase budget. The architecture doesn't rely on them —
the model has no database access regardless of what it's persuaded to say.

---

## 15 — Session isolation

Concurrent users must never retrieve each other's uploads.

- Unguessable IDs (`secrets.token_urlsafe(32)`, 256 bits) — stops *guessing*
- **Mandatory filtering in the store** — is the actual isolation

> There is deliberately **no code path** that reaches the session collection without a
> filter. Default with no filter is **zero** documents, never all.

_"The failure mode I designed against is a caller who forgets. So I removed the ability to
forget."_

---

## 16 — On-premises deployment

| | |
|---|---|
| **Torch-free CPU profile** | CTranslate2 + ONNX + llama.cpp → **~1.5 GB**, not 4.5 GB |
| **Two profiles, one codebase** | Device detection, never a code fork |
| **Offline install** | `uv pip download` + `fetch_models.py` → air-gapped, nothing touches the network |
| **Docker** | 3 services, one image, model weights as a volume |
| **Reproducible** | `pyproject.toml` + committed lockfile |

**Every core model is commercially licensed** — Apache-2.0 or MIT.

---

## 17 — A licensing finding

`jina-embeddings-v3` is the strongest multilingual embedder in the shortlist.
**It is CC-BY-NC-4.0. Non-commercial.**

Same for `jina-reranker-v2` and `f5-tts-egyptian-arabic`.

**Replaced with `BAAI/bge-m3` (MIT)** — which also has ~20× the adoption. Zero cost to
switch.

_"A non-commercial model in a telecom's customer-facing assistant fails review for reasons
that have nothing to do with benchmark scores. Licence is a hard filter, checked before
quality."_

---

## 18 — Evaluation, and what I did *not* take on trust

`Nawah-ASR-118M-v5` claims WER **0.3358 vs Whisper large-v3's 0.4149** on Egyptian Arabic.

- **43 downloads/month, 2 likes**
- The WER is **self-reported on the author's own eval set**
- Not Whisper-architecture → needs torch → **cannot run under CTranslate2**, so it's out of
  the torch-free CPU profile on engineering grounds before quality is even discussed

**Decision: every model slot is pluggable; well-adopted defaults ship; the Egyptian
candidates get measured.** That converts an unverified claim into an evaluation result
instead of a gamble on the critical path.

---

## 19 — Scalability: what's built and what isn't

**Built:** stateless request handling · singleton model loading · serialised llama.cpp
access · WAL SQLite for concurrent readers · session TTL and eviction · lazy model
loading for the 16 GB VRAM ceiling

**Honest limit:** the session store is **in-process**. A second `core` replica would break
session affinity today. The interface is narrow — Redis is a one-file swap — but horizontal
scaling is currently a **design property, not a demonstrated one.**

_"I'd rather say that than let someone find it by scaling the container."_

---

## 20 — Roadmap

| Next | Why not now |
|---|---|
| BGE-M3 on GPU | Should close much of the 73% dialect gap |
| Redis sessions | Makes horizontal scaling real |
| Cross-encoder reranking on CPU | Too slow on 4 cores today |
| LoRA fine-tune on telecom QA | Real, but not a POC-scale task |
| Langfuse tracing | Needs a server; JSONL tracing for now |
| Per-document auth, visual retrieval | Deliberately deferred |

---

## Closing

> The core runs offline on a CPU. Every answer is cited. The model detects intent and
> nothing else — the backend decides what may happen. And the numbers on these slides came
> out of `/metrics` on the running system.

---

## Appendix (hold in reserve for questions)

- Per-stage latency breakdown, both profiles
- Chunking: heading-aware, overlap, the merge-vs-drop bug
- `llama-cpp-python` is sdist-only on PyPI → source build → fails without MSVC and on
  Windows' 260-char path limit. Pinned wheel index; Python 3.11 not 3.13.
- fastembed defaulted to 1 ONNX thread → 0.8 chunks/s. Pinned to physical cores: **39 s
  instead of 12 minutes.**
- The 7 retrieval misses — all pages that *are* in the corpus but rank below 5, i.e. an
  embedder limit, not a coverage gap
