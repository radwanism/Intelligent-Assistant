# ASR evaluation clips

`scripts/eval_asr.py` measures word error rate on the clips listed in `manifest.jsonl`.
The harness is complete; **the clips are not recorded yet, so there is no WER figure.**
Recording them is about twenty minutes on a phone.

## What to record

The case study is about *noisy* audio and *Egyptian dialect*, so a manifest that cannot
speak to both is not worth running. Aim for roughly eight clips:

| # | Dialect (`dialect`) | Condition (`condition`) | Say |
|---|---|---|---|
| 1–2 | `arz` Egyptian | `clean` | a plain telecom question, e.g. «عايز أعرف أسعار باقات الإنترنت» |
| 3–4 | `arz` Egyptian | `noisy` | the same kind of question with traffic, a fan, or a TV on |
| 5 | `ar` MSA | `clean` | «ما هي باقات الإنترنت المنزلي المتاحة؟» |
| 6 | `en` English | `clean` | "How do I recharge my line?" |
| 7 | `en` or `arz` | `noisy` | any question, low volume, with background speech |
| 8 | — | `silence` | three seconds of room tone, no speech (exercises the hallucination filter) |

Record as mono WAV (16 kHz or higher). Write the **reference exactly as spoken** —
the dialect, not a cleaned-up MSA version — otherwise the score measures your editing.

## The manifest

Copy `manifest.example.jsonl` to `manifest.jsonl`, replace the lines with your clips, and
put the audio files in this directory. One JSON object per line:

```json
{"audio": "clip01.wav", "reference": "عايز أعرف أسعار باقات الإنترنت", "dialect": "arz", "condition": "clean"}
```

## Run

```bash
python scripts/eval_asr.py --check     # validate the manifest and its coverage — no model loaded
python scripts/eval_asr.py             # transcribe and score
```

`--check` fails on a missing file or field and *warns* when a condition or dialect has no
clip, because a manifest without a noisy clip cannot produce a noisy-audio WER.

Audio files (`*.wav`, `*.mp3`, `*.m4a`) are git-ignored; the manifest is not.
