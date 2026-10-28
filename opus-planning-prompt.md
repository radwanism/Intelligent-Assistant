# System Planning Brief — Telecom Egypt Intelligent Assistant (Case Study)

## Your Role

You are acting as a **senior software architect**. Your job is to help me **plan and design** a complete implementation of the case study below — collaboratively, as a planning conversation, not a one-shot document dump. Where information is insufficient or ambiguous, ask me clarifying questions before finalizing that part of the design, rather than guessing.

Read every requirement below carefully — both the **official case study** (Section A) and **my own additional architecture constraints** (Section B). Nothing should be skipped, generalized, or silently dropped. If you believe a requirement in Section A conflicts with one in Section B, say so explicitly instead of resolving it silently — flag it and let's decide together.

---

# SECTION A — Official Case Study (from Telecom Egypt's assignment PDF)

*The wording below is the assignment's. The structuring into headings/tables is mine, and anything in italic parentheses is my own note, not the client's text.*

## A.0 Title
**Telecom Egypt — Intelligent Assistant**

## A.1 Overview
Build a **production-ready** intelligent assistant for Telecom Egypt that transforms customer service interactions (voice, text, documents) into structured outputs and actionable insights. It must handle real-world conditions including **noisy recordings** and **Egyptian dialect variations**, delivering **bilingual outputs (Arabic and English)**.

## A.2 Business Scenario
Telecom Egypt receives thousands of customer inquiries daily. The company wants an AI-powered assistant that can:

- Answer customer questions using the **Telecom Egypt official website (te.eg)** resources.
- Handle queries in **Arabic, English, and Egyptian dialect** — via voice or text.
- **Detect whether input is voice or text** and respond accordingly. *(This is known from the UI channel the request arrives on — it does not call for an input-type classifier.)*
- **Speak the answer back** when voice-based; **always display text alongside the voice response** (never voice-only).
- Allow users to **upload and query documents** (PDF, DOCX, TXT, Images) — uploaded content must be **embedded and included in the retrieval/search context** for response generation.
- **Ground all responses** in Telecom Egypt resources, and provide **source citations** for every answer.
- Display **chat history**.
- Provide **audio playback** for voice responses.

## A.3 Knowledge Sources

| Source | Description |
|---|---|
| te.eg website | Official Telecom Egypt website — **primary knowledge base** |
| User Documents | PDF, DOCX, TXT, Image formats uploaded by the user |
| Voice Input | Live speech captured via microphone |

## A.4 Interaction Modes

| Input Mode | Scenario | Response |
|---|---|---|
| Voice input | User speaks in Arabic, English, or Egyptian dialect. ASR converts speech to text, passed to the RAG pipeline. | Synthesized voice (TTS) **+** text in chat |
| Text input | User types a question in the chat interface. | Textual response including relevant citations |

## A.5 Required Deliverables

| Deliverable | Description |
|---|---|
| Working Interface | Streamlit or Gradio app with voice input, text chat, document upload, and source citations |
| Working Solution | End-to-end pipeline for **on-premises execution** |
| Python Notebook | Complete code with explanations covering the **ASR and RAG pipeline** |
| Presentation | PPT slides explaining system architecture, workflow, and on-premises deployment |
| Documentation | README with setup instructions, including **how to run locally** |

## A.6 Evaluation Notes (from the assignment itself)

1. This case study evaluates **practical RAG and ASR implementation capabilities** through a **Proof of Concept (POC)**.
2. Candidates are expected to demonstrate their **technical approach, problem-solving abilities, system design thinking, and on-premises deployment considerations** during the presentation.
3. **External APIs may be used for benchmarking or comparison purposes only** — **not as the core solution.** *(My reading: the core solution must therefore stay local / on-premises / open-source. See B.3.)*
4. The requested scope is the **baseline**; candidates are **encouraged to showcase creativity and propose additional value-added features** where applicable.

---

# SECTION B — My Own Additional Architecture Decisions & Constraints

The case study (Section A) explicitly invites "additional value-added features" (A.6.4) as long as the core stays on-premises (A.6.3). The following are constraints and creative additions I want layered on top of the baseline, to demonstrate engineering depth. These are **not** optional extras to skip — treat them as required parts of the plan, scoped to the build window in Section B.2.

### B.1 What I Need From You
- Review all the features, data flow, and requirements from Section A and this Section B together.
- Produce a plan for the **actual data flow** across the whole system.
- Suggest the most relevant technical approach needed — **without over-engineering**.
- Clarify all the steps in detail needed to build this from **A to Z**.
- Split the system into **components** using system-design principles, and explain the **integration between each component**.
- Identify where a component genuinely needs to be its **own service** so a failure in one feature doesn't take down others — and justify each case (don't split by default).

### B.2 Complexity Boundary — Balance, Don't Over-Engineer
**Hard constraint: I am building and delivering this POC in a 12-hour window, on the CPU-only hardware in B.3.** It is meant to demonstrate engineering ability — not a production system at scale, despite the "production-ready" language in A.1.

- **Every requirement in Sections A and B must actually be implemented**, not merely described. The time box governs *how deeply* each one is built, never *whether* it is built.
- For each requirement, choose the smallest implementation that genuinely demonstrates the concept **end to end and verifiably in a live demo**. A stub that cannot be demonstrated does not count as implemented.
- **Do not**, in the process of simplifying, miss or omit any technical concept explicitly mentioned in Sections A or B. Balance simplicity against completeness — every concept listed here must appear *somewhere* in the plan, even if scoped down.
- If a particular approach would clearly be over-engineering given the 12-hour window, **do not build it into the main plan** — put it in a dedicated **"Future Work"** section instead, with a short note on why it's deferred.
  - **Example given explicitly by me:** per-document user authentication (i.e., authenticating a user against ownership of specific uploaded documents) belongs in Future Work, not the core build.
- Because the window is this tight, **build order matters as much as the design** (see D.16). Be explicit about what must be working before what, where the cut lines are if something runs long, and which 15 minutes of the demo prove the most.
- If you judge that the full A+B scope genuinely does not fit in 12 hours on CPU-only hardware, **say so directly and tell me what you'd cut first** — do not silently thin out a requirement to make the timeline work.

### B.3 Deployment & Runtime Constraints
- Prepare a guideline to **run everything locally** as the core solution. Per A.6.3, external APIs are allowed only for benchmarking/comparison — never as the core dependency.
- **Target hardware — two profiles, one codebase.** The on-premises claim is CPU-only, so that is the *required baseline*; a GPU may only make things faster, never make them work.
  - **`cpu-lite` (baseline, and what the on-prem deliverable claims).** Every model in the core path — ASR, embeddings, LLM, TTS — must run acceptably on CPU with **no GPU present**. Quantized/optimized runtimes only (int8 CTranslate2 ASR, a small quantized LLM via llama.cpp, ONNX embeddings, a lightweight TTS). Prefer a torch-free stack; it keeps the install small enough to be a credible on-prem artifact. This is the same constraint as the fully-offline CPU-only pipeline in C.1.
  - **`gpu-colab` (development/demo runtime).** Development happens on a CPU-only laptop, but the system is pushed to GitHub and run on Google Colab (T4, 16 GB VRAM) for testing and the demo, so larger models are available there. Selection is by **runtime device detection**, never by a code fork.
  - For **each** model choice, state its footprint and realistic latency **on both profiles**, and **do not propose anything that silently assumes a GPU.** Report CPU numbers as the on-premises result.
  - Mind the VRAM ceiling: the full GPU stack will not fit resident in 16 GB. Say which models stay loaded and which lazy-load and unload. Note also that a T4 is Turing — **fp16 only, no bf16, no flash-attention**.
- **Licensing is a hard filter.** Every model in the **core** path must carry a license that permits commercial use (Apache-2.0, MIT, and similar). **CC-BY-NC and other non-commercial licenses are disqualified as core components** regardless of benchmark scores — a non-commercial model in a telecom's customer-facing assistant fails review for reasons unrelated to engineering. Such models may still be used strictly for benchmarking, per A.6.3. State the license next to every model you propose.
- **te.eg knowledge base acquisition** — specify how te.eg content is actually obtained: crawl scope and depth, how the Arabic *and* English versions of the site are both covered, JS-rendered pages, robots.txt/ToS, target page count, and whether the POC ships a one-time snapshot or re-crawls. Given the 12-hour window a bounded snapshot is expected — say how large and how it is built, since retrieval quality is capped by this step.
- **Cache user-uploaded files per session**, so the user can ask questions grounded in what they uploaded during that session. (This directly implements the document-upload requirement in A.2/A.3 — cover how uploaded PDFs, DOCX, TXT, and **images** get parsed/OCR'd, embedded, and scoped to that session.)

### B.4 LLM ↔ Database Safety Architecture (Critical — do not simplify away)
The POC **must include at least two action-shaped intents** backed by a small local database (SQLite is fine): one read — e.g. `check_bill_balance` — and one write — e.g. `create_support_ticket`. This is mandatory because Section A on its own never forces a DB action (te.eg RAG plus session documents is entirely read-only), and without a real action this whole requirement collapses into prose. The chain below must be **demonstrated running**, not described.

Given such an action, **do not let the LLM connect to or act on the DB directly.** The required flow is:

1. **Input guardrails** run first, before any request reaches the model — to catch jailbreak attempts and prompt injection.
2. **PII masking** runs next — sensitive data (emails, bank card numbers, etc.) must **never be visible to the model** directly.
3. The LLM's only job in this flow is to **detect intent** — it never performs the action itself (e.g., it never writes to a DB).
4. The **backend** is responsible for checking **user permissions** and actually performing the requested action, only if the user is authorized.

This separation of "LLM detects intent" vs. "backend authorizes and executes" is a core architectural requirement, not a nice-to-have — it's exactly the kind of design-thinking depth the evaluation notes (A.6.2) call out. Include the negative cases in the demo: a prompt-injection attempt that the guardrail stops, and an unauthorized action that the backend refuses **even though the LLM correctly detected the intent**.

### B.5 Model Selection
Open-source Arabic-capable LLM candidates to consider (for reference — see note below):
`ALLaM`, `Fanar`, `Jais`, `SILMA`, `Mistral Saba`, `Cohere Arabic`, `Qwen`, `Gemma`, `Llama`

**Important:** The focus of this design should **not** primarily be *which* model is chosen. The focus should be on the surrounding engineering concepts and implementing what genuinely makes sense for this task, across:

- **Retrieval**
- **Data quality**
- **System design**
- **Evaluation**
- **Scalability**

Whichever model is chosen, justify its size and quantization against the CPU-only constraint (B.3) and the latency budget (B.10) — not against benchmark leaderboards.

### B.6 User Frustration Detection → Human Handoff
Design a mechanism to detect user frustration during interaction, with a path to **transfer the conversation to a human agent** when needed.
- Prefer cheap explicit signals (repeated rephrasing of the same question, consecutive retrieval misses, low answer confidence, direct escalation requests) combined with a lightweight Arabic sentiment classifier — the MARBERT/MARBERTv2 line in C.4 is directly relevant and far cheaper on CPU than an LLM judgment call on every turn.
- Define the **handoff contract**, not just the trigger: what conversation state and context transfers to the human agent, what the user sees at the moment of transfer, and what happens if no agent is available.

### B.7 Testing
- Prepare **unit tests** covering all services and their functionality.
- At minimum, the guardrail chain (B.4), PII masking, and session isolation (B.8) need tests that assert the *negative* cases — these are security properties, and a test that only checks the happy path proves nothing about them.

### B.8 Session & Concurrency Security
- Ensure **session tokens are secure**, specifically to handle **concurrent users** using the assistant at the same time — this must prevent **retrieval cross-contamination** (one user's session must never retrieve from another user's uploaded documents).
- Concretely: the POC has **no login** (per B.2, user auth is Future Work), so use anonymous, cryptographically random session IDs. Isolation must be *enforced* — every retrieval call carries a mandatory session-scoped metadata filter, and the default with no filter is "no user documents", never "all user documents". Do not rely on callers remembering to pass the filter.
- Say how session data is expired and evicted, and what happens to uploaded files and their embeddings at session end.

### B.9 Tooling
- Use **`pyproject.toml`** with **`uv`** for dependency/project management instead of `requirements.txt`. Commit `uv.lock` — exact, reproducible installs are the whole point for an on-premises deliverable. Note how the install works on a machine with no internet access.

### B.10 Success Criteria & Latency Budget
Do not leave "Evaluation" (B.5) as an abstract pillar — the plan needs numbers to design against, because on CPU-only hardware every model-size decision is a latency trade-off.

- Build a small **bilingual golden set** (~30–50 Q/A pairs over te.eg content, spanning MSA, English, and Egyptian dialect phrasings) and a handful of noisy/dialect audio clips.
- Define target metrics and state what "good enough for this POC" is for each: **WER** on the Egyptian-dialect clips, **retrieval recall@k**, **citation correctness** (does the cited source actually support the answer), and **refusal/groundedness** rate on out-of-scope questions.
- Set an explicit **p95 end-to-end latency budget** for voice→voice and text→text on CPU, and break it down per stage (ASR / retrieval / generation / TTS). Every model choice in B.5 must be argued against this budget.
- A WER-and-latency-per-session monitoring layer like the one in C.1 is the right shape here and is cheap to add.

### B.11 Structured Outputs & Insights (implements A.1)
A.1 promises the assistant "transforms customer service interactions into **structured outputs and actionable insights**" — this must not get lost behind the chat UI.

- Emit a **structured record per turn**: detected intent, extracted entities, language/dialect, sentiment/frustration score, whether the answer was grounded, citations used, and resolution status.
- Surface a small **session-level insights view** over those records (top intents, escalation rate, unanswered questions, average latency).
- This is nearly free: it reuses the same constrained-decoding intent call from B.4 and the sentiment signal from B.6. It also doubles as the demo's evidence that the pipeline is doing what you claim.

---

# SECTION C — Company Context (Grounding Only — not literal requirements)

The following are real excerpts from AI engineering job descriptions/experience at Telecom Egypt. Use them to understand what the company actually works on, and to decide which techniques in the design would be most relevant and land well for this audience. **These are context for scoping relevance and demonstrating awareness of the company's stack — not a literal requirements list to copy wholesale into the POC.**

To be explicit about what to draw on and what to leave alone: **directly relevant** are C.1 (CPU-only offline STT/TTS, int8 quantization, Egyptian-dialect tuning, WER/latency monitoring, hallucination filters), C.2 (RAG over a local vector store, intent-based routing, guardrails before customer-facing actions, tracing/eval instrumentation), C.3/C.6 (Arabic LLM work, local ASR), and C.4's Arabic NLP line (MARBERT for sentiment/NER). **Not relevant** — do not shoehorn these in — are the vision/facility-monitoring work (YOLO, EfficientNet, fire–smoke detection, face authorization), churn prediction, and OCR captcha work. Fine-tuning (C.3) is real experience but almost certainly does not fit a 12-hour window; if you agree, it belongs in Future Work.

### C.1 — Real-Time Speech-to-Speech Translation Platform
> Architected a bi-directional Arabic/English translation pipeline (STT, MT, TTS) that runs fully offline on CPU-only hardware, keeping all customer audio on-premise. Cut end-to-end inference latency by roughly 30% with Faster-Whisper, CTranslate2, and Int8 quantization, tuned specifically for Egyptian Arabic dialects. Built asynchronous audio streaming on Python, FastAPI, and WebSockets; containerized services with Docker for reproducible on-premise deployment. Added regex-based hallucination filters and a monitoring layer tracking Word Error Rate and latency per session.

### C.2 — Multi-Agent Customer Intelligence & Campaign System
> Designed a multi-agent platform that analyzes subscriber data and routes each customer to a specialized agent (billing, technical support, retention, upsell) based on detected intent. Orchestrated agent workflows with LangGraph and CrewAI; exposed internal tools and data sources to agents through MCP (Model Context Protocol) servers and structured function calling. Grounded agent responses in telecom knowledge bases via RAG over a ChromaDB vector store, with output guardrails applied before any customer-facing action. Instrumented the full pipeline with LangSmith and Langfuse for agent tracing, prompt evaluation, and regression testing before campaign rollouts.

### C.3 — Arabic LLM Fine-Tuning
> Fine-tuned open-source LLMs using LoRA and QLoRA (PEFT) for telecom customer-support tasks, improving response quality on Egyptian Arabic queries while keeping training cost low.

### C.4 — Broader ML Engineering Role
> Develop and fine-tune Transformer-based Arabic NLP models (MARBERT, MARBERTv2) for sentiment analysis, NER, and large-scale text classification. Build scalable ML pipelines and deploy models via FastAPI microservices integrated with Vue.js production dashboards. Implement RAG pipelines and LLM Agent–based automation using CrewAI and LangGraph to support internal analytics and decision workflows. Contribute to the Integrated Smart Facility Monitoring System, deploying real-time vision models (YOLO, EfficientNet, Triplet-Loss CNN) for fire–smoke detection and face authorization, and exposing services through FastAPI. Develop interactive prototypes using Streamlit and maintain high-quality ML codebases aligned with MLOps best practices. Collaborate with cross-functional teams to convert business requirements into fully deployable AI features.

### C.5 — Leadership Scope
> Leading the AI Applications Development and Data Analytics Research section, driving innovation through the design, development, and implementation of AI-powered solutions and advanced analytics. Overseeing research in machine learning, deep learning, big data analytics, and emerging AI technologies to improve services, enhance decision-making, and boost operational efficiency. Managing a skilled team, fostering cross-functional collaboration, and aligning projects with Telecom Egypt's digital transformation strategy.

### C.6 — Current ML Engineer Role
> In my current role as a Machine Learning Engineer at Telecom Egypt, I have successfully developed and deployed models for: Arabic Agentic RAG Chatbot using local LLMs; Speaker-Diarization and speech analysis of customer service calls using MFCC & RNNs; Automatic Speech Recognition (ASR) local portal with 60% improvement in Arabic; Fine-tuned LLaMA 3.1 on telecom QA using LoRA and SFTTrainer which improved inference by 72%; Customer Churn Prediction; OCR captcha identification for web scraping; Responsible for analyzing the GPU requirements, resources and prices between Gen-AI models.

---

# SECTION D — Expected Output Structure From You

When you produce the plan, structure it to explicitly cover:

1. **Requirements review** — a confirmation pass showing you've captured every item in Sections A and B (ask questions here if anything is unclear).
2. **System architecture / component breakdown** — per system-design principles, with integration between components explained. Must account for: the **te.eg crawl & ingestion pipeline**, ASR ingestion, text ingestion, document ingestion (PDF/DOCX/TXT/**Image**) with embedding, the RAG retrieval layer over te.eg content + user documents, response generation with citations, TTS, and the chat/voice UI.
3. **Data flow plan** — the full journey of a voice or text request (and of a document upload) through the system, end to end.
4. **Service-boundary decisions** — where something becomes its own service, and why (failure isolation).
5. **Local/on-premises deployment guide** — open-source stack, how to run locally, noting where (if anywhere) an external API is used strictly for benchmarking per A.6.3.
6. **Security & guardrail architecture** — input guardrails → PII masking → intent-only LLM → permission-checked backend execution (B.4).
7. **Session handling** — per-session document caching/embedding scope, secure concurrent-session token isolation to prevent cross-document retrieval (B.3, B.8).
8. **Core pillars coverage** — explicit treatment of Retrieval, Data quality, System design, Evaluation, and Scalability (B.5), with the concrete metrics and latency budget from B.10.
9. **Bilingual & dialect handling** — how Arabic, English, and Egyptian dialect are handled across ASR, retrieval, and generation, and how noisy audio is handled (A.1, A.2). Cover the cross-lingual retrieval problem specifically: an Egyptian-dialect spoken question hitting an MSA or English source page.
10. **Structured outputs & insights** — the per-turn structured record and session insights view (B.11), and how they are produced without a second model pass.
11. **Frustration detection → human handoff** design (B.6), including the handoff contract.
12. **Testing strategy** — unit tests per service, with negative-case tests for the security properties (B.7).
13. **Tooling** — `pyproject.toml` + `uv` project setup, including offline install (B.9).
14. **Deliverables mapping** — explicitly map the plan to the 5 required deliverables in A.5 (Streamlit/Gradio interface, working on-premises solution, Python notebook covering ASR+RAG, PPT slide outline, README). For each, say whether it is produced as a working artifact or as an outline.
15. **Future Work** — anything that would be over-engineering for a 12-hour build (e.g., per-document user authentication), clearly separated from the core plan, each with a one-line reason.
16. **Build order** — a concrete hour-by-hour sequence for the 12-hour window: what must work before what, the earliest point at which there is a demo-able end-to-end path, and the cut lines if something runs long (B.2).

---

## How to work with me

Work in **two phases, and stop between them** — I want a planning conversation, not a one-shot document dump:

- **Phase 1 — confirm & question.** Produce *only* the requirements-confirmation pass (item 1) plus your clarifying questions and any A-vs-B conflicts you've spotted. Do not write the architecture yet. Given the 12-hour window, restrict questions to ones whose answers would actually change the design.
- **Phase 2 — the plan.** After I answer, produce the full structure above.
