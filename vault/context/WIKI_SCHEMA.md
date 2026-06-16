# Wiki Schema

This file defines the structure and conventions for the project knowledge wiki.
Read it before making any wiki change. (Bootstrapped 2026-06-16.)

## Directory layout

| Path | Purpose | Editable |
|------|---------|----------|
| `vault/context/index.md` | Content catalog — one line per page. Read first to find pages. | yes |
| `vault/context/log.md` | Append-only activity log. One entry per wiki change. | append-only |
| `vault/context/WIKI_SCHEMA.md` | This file. Conventions + rules. | rarely |
| `vault/context/wiki/` | All wiki pages. Owned entirely by the wiki maintainer. | yes |
| `vault/context/sources/`, `vault/context/raw/` | Immutable source material. | read-only |

## Page conventions

- **Filenames:** `kebab-case.md`. Concept pages prefixed `concept-`, guide pages `guide-`.
- **Frontmatter** (YAML) on every page:
  ```yaml
  ---
  title: Human Readable Title
  category: concept | guide | reference
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  tags: [tag1, tag2]
  ---
  ```
- **Cross-references:** use `[[wikilinks]]` (no `.md` extension), e.g. `[[concept-urbanspeech]]`.
- **Style:** prefer tables and bullets over prose. Document architecture, not every commit.

## When to update

- After a structural change (new dagspace, stage, config group, renamed module).
- After adding/removing/significantly modifying a pipeline stage or shared utility.
- After resolving a documented bug or performance issue.
- When asked to document something, or when a page is discovered stale.

Do **not** update for trivial changes (typos, comment edits, test-only changes).

## Update procedure

1. Read `index.md` to find the relevant page(s).
2. Read the page(s) and the current source.
3. Edit the page; bump the `updated:` date.
4. Fix/add `[[wikilinks]]`.
5. Update `index.md` if scope changed or a page was created.
6. Append an entry to `log.md`: `## [YYYY-MM-DD] action | Subject`.
