# Data notice — read before making this repository public

A pre-publication audit of everything tracked by git. Short version: **no credentials,
no personal data, no machine paths.** One item needs a decision from you.

---

## ⚠️ The one thing to decide: the te.eg content

The repository commits a **snapshot of Telecom Egypt's public website**:

| File | Size | Contents |
|---|---|---|
| `data/corpus/te_eg.jsonl` | 738 KB | 388 cleaned pages, ~755,000 characters |
| `data/chroma/` | ~10 MB | embeddings + the same text stored alongside them |

It is committed deliberately — it is what lets a fresh clone (or Colab) answer questions
without re-crawling. But it means **publishing the repo republishes Telecom Egypt's
website content**.

**`robots.txt` permits crawling. That is not the same as permission to redistribute.**
`https://te.eg/robots.txt` is `User-Agent: * / Disallow:`, which governs what a crawler may
*fetch*. Copyright in the page content is unaffected, and te.eg's terms may restrict
redistribution independently.

There is also a non-legal consideration: this is a case study **for Telecom Egypt**.
Publishing a scrape of their site in a public job-application repository is a bad look
even where it is lawful.

### Options

| Option | Effect | |
|---|---|---|
| **Private repository** | Nothing changes; clone works as-is | ✅ **Recommended** |
| Public, corpus removed | Add `data/corpus/` + `data/chroma/` to `.gitignore`; users run `te-ingest` themselves (~15 min) | Breaks the one-click Colab path |
| Public, corpus kept | Only with Telecom Egypt's agreement | Ask first |

**For Colab from a private repo**, clone with a fine-grained personal access token scoped
to this one repository, read-only:

```python
from getpass import getpass
token = getpass("GitHub token: ")          # never hard-code it in a cell
!git clone https://{token}@github.com/USER/te-assistant.git
```

To strip the corpus instead:

```bash
git rm -r --cached data/corpus data/chroma
printf 'data/corpus/\ndata/chroma/\n' >> .gitignore
git commit -m "Remove te.eg snapshot from version control"
```

> Note this only affects future commits. The snapshot stays in git history — if the repo
> was ever pushed publicly, rewrite history (`git filter-repo`) or start a fresh repo.

---

## ✅ Everything else is clear

### No credentials or secrets
Scanned all tracked files for API keys, tokens, passwords, `hf_*` / `sk-*` patterns and
bearer tokens. **Nothing found.** The system needs no API key — that is the point of it.

The only `username`/`password` strings in the corpus are te.eg's own **published** router
setup instructions ("Enter the default username: admin"), already public on their site.

### No machine paths or local identity
No `C:\Users\...`, no home directories, no developer usernames in any tracked file.

### Commit identity
Commits are authored as `Radwan <33943019+radwanism@users.noreply.github.com>` — GitHub's
noreply address. **No personal email is exposed.**

### The case-study brief and CV material are NOT in the repository
`opus-planning-prompt.md`, `use-case.txt` and `TelecomEgypt_AI_Case_Study.pdf` live in the
parent directory and were never added. This matters: that brief contains job-description
and CV excerpts. **Keep it that way** — do not copy them in.

### Demo data is synthetic
`SEED_CUSTOMERS` in `actions/db.py`:

| Field | Value | |
|---|---|---|
| Names | "Mona Adel", "Karim Fouad" | invented |
| Numbers | `01012345678`, `01198765432` | sequential placeholders |
| Balances | 213.50, −42.00 | invented |

Test fixtures follow the same pattern — `4111111111111111` is the standard Visa *test*
card, `mona.adel@example.com` uses the RFC 2606 reserved domain. The national ID and IBAN
in `test_security.py` are structurally shaped but not real.

**These exist to prove PII masking works.** A masking test with no PII-shaped input proves
nothing.

### Corporate contact details in the corpus
Three addresses appear, all **published by te.eg** as public contact points:
`Customer.care@te.eg`, `Corp.Sutainability@te.eg`, `B.S@te.eg`, plus one hotline number.
Public corporate contacts, not personal data.

### Runtime data is excluded
`.gitignore` keeps out: `data/sessions/` (customer uploads), `data/traces.jsonl` (turn
records), `data/assistant.db` (tickets), `data/corpus/raw.jsonl` (150 MB of raw HTML), the
venv, and all model weights.

Verified: `raw.jsonl` and `smart_assistant/` are **not** staged. Largest tracked file is
`data/chroma/chroma.sqlite3` at 10 MB — well within GitHub limits.

---

## Before you push

- [ ] Decide public vs private (see the te.eg question above)
- [ ] `git log --format='%ae' | sort -u` — confirm no personal email
- [ ] Set `REPO_URL` in `notebooks/te_assistant_walkthrough.ipynb` cell 1
- [ ] `git status --short` is empty
- [ ] Confirm the brief and CV files are still outside the repository
