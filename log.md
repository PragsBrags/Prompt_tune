### Design Choices

# Design Rationale: CoT, Back-Translation, and Iterative Review

## Why "silent" reasoning was removed (CoT)

Chain-of-thought works because of autoregression: each generated token conditions the next, so intermediate reasoning tokens are extra *computation* the model gets to use before answering. Asking a standard (non-reasoning) LLM to reason "silently" gives it no channel to do that — analysis and answer collapse into one forward pass, so nothing is actually decomposed. The fix: make every step (tense → polarity → sentence type → subject-object structure → key terms → draft → check) literal, visible output, then parse it out.

**Sources:**
- Wei et al., 2022 — original CoT paper; notably never tested MT empirically,
  only flagged it as a plausible future direction.
- He et al., 2023 (**MAPS**) — mines keywords/topics, generates exemplars as
  visible steps before producing translation candidates.
- Lu et al., 2023 (**Chain-of-Dictionary**) — extracts and writes out
  dictionary lookups before translating.

## Why back-translation is structured this way

Round-trip translation as a quality check predates LLMs — a benchmark-free way to estimate quality without a reference translation, comparing the source against itself. The LLM-era framing: an ideal translation should let a capabl LLM recover the original sentence, so the original-vs-back-translation gap
becomes an implicit quality signal.

Key design choice — explicitly telling the reviewer that wording differences are expected and not automatic errors — exists because **back-translation mismatch ≠ forward-translation error**. Translation isn't one-to-one; a correct translation can back-translate into different, still-correct phrasing from synonym/paraphrase choice alone.

**Sources:**
- van Zaanen & Zwarts, 2006 — round-trip translation as a benchmark-free
  quality checker.
- Wangni, 2024, *"Language Models and Cycle Consistency for Self-Reflective MT"*
  — cycle consistency between original and back-translation as an implicit
  quality estimate.
