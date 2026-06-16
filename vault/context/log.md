# Wiki Activity Log

Append-only. One entry per wiki change: `## [YYYY-MM-DD] action | Subject`.

## [2026-06-16] bootstrap | Wiki scaffolded
- Created `WIKI_SCHEMA.md`, `index.md`, `log.md` (the wiki did not previously exist on disk despite the CLAUDE.md reference).
- Added `wiki/concept-dagspaces.md` — dagspace architecture overview.
- Added `wiki/concept-urbanspeech.md` — urbanspeech pipeline + VAD-gated ASR hallucination control (Silero VAD in extract_audio, segment-driven chunking + post-filter in asr).
