# Wiki Schema

This document defines conventions, structure, and workflows for maintaining the MLLMSCI project wiki. It is the configuration file that makes the LLM a disciplined wiki maintainer.

## Vault Structure

```
vault/context/
  WIKI_SCHEMA.md    ← this file (schema / conventions)
  index.md          ← content catalog (LLM-maintained)
  log.md            ← chronological activity log (append-only)
  wiki/             ← LLM-generated markdown pages
  sources/          ← curated source documents (immutable)
  raw/              ← raw data files, images, exports (immutable)
```

## Layers

### Sources (`sources/`, `raw/`)
- Immutable. The LLM reads but never modifies these.
- Place articles, papers, exported configs, data samples here.
- Each source should be referenced by at least one wiki page.

### Wiki (`wiki/`)
- Entirely LLM-owned. The LLM creates, updates, and deletes pages.
- Every page uses the frontmatter template below.
- Cross-references use Obsidian `[[wikilinks]]` syntax.

### Schema (`WIKI_SCHEMA.md`)
- Co-evolved by human and LLM as conventions solidify.
- Any structural change to the wiki should be reflected here first.

## Page Frontmatter Template

```markdown
---
title: Page Title
category: overview | architecture | dagspace | infrastructure | config | guide | concept | reference | troubleshooting
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [tag1, tag2]
sources: [source_file.md, ...]
---
```

## Categories

| Category | Purpose | Examples |
|----------|---------|---------|
| `overview` | High-level project summaries | Project overview, getting started |
| `architecture` | System design and patterns | Pipeline execution model, DAG engine |
| `dagspace` | Individual dagspace documentation | UrbanVQA, UrbanOCR, etc. |
| `infrastructure` | Shared modules and utilities | vLLM inference, Ray setup, W&B logger |
| `config` | Configuration system docs | Hydra composition, model configs, SLURM launchers |
| `guide` | How-to guides for users | Bootstrapping, adding a new dagspace, custom stages |
| `concept` | Domain concepts and definitions | Counterbalancing, tiling, guided decoding |
| `reference` | Quick-reference tables and lookups | CLI commands, config options, file locations |
| `troubleshooting` | Known issues and fixes | Batch size collapse, object store growth |

## Naming Conventions

- Filenames: `kebab-case.md` (e.g., `urban-vqa.md`, `pipeline-execution-model.md`)
- Use descriptive names, not abbreviations
- Prefix concept pages with `concept-` (e.g., `concept-counterbalancing.md`)
- Prefix guide pages with `guide-` (e.g., `guide-bootstrapping.md`)

## Cross-Referencing

- Use Obsidian `[[page-name]]` links (no `.md` extension)
- Use `[[page-name#heading]]` for section links
- Every page should link to at least one other page
- Overview and architecture pages should be richly linked

## Workflows

### Ingest New Source
1. Place source in `sources/` or `raw/`
2. Create or update relevant wiki pages
3. Update `index.md` with new/changed pages
4. Append entry to `log.md`

### Answer a Query
1. Read `index.md` to find relevant pages
2. Read those pages for context
3. Answer the query
4. If the answer reveals a gap, create a new page

### Maintenance Pass
1. Scan `index.md` for stale or thin pages
2. Re-read source material and codebase
3. Update pages with new information
4. Remove pages for deleted code
5. Log the maintenance pass

## Log Format

Each entry in `log.md` starts with a parseable header:

```
## [YYYY-MM-DD] action | Subject
```

Actions: `ingest`, `create`, `update`, `query`, `maintenance`, `bootstrap`

This format enables: `grep "^## \[" log.md | tail -5`

## Quality Standards

- Every wiki page should be self-contained enough for a newcomer to understand without reading other pages first
- Code examples should be real (from the codebase), not hypothetical
- File paths should be relative to project root
- Keep pages focused: one concept or component per page
- Prefer tables and bullet points over prose for reference material
