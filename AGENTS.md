# Distributed Video DiT Inference — Learning Reference Library

## Project purpose

This repo is a **grounded reference library** for building a pipeline-parallel inference system for a video Diffusion Transformer. It serves two consumers:

1. **A human learner** studying the material topic-by-topic
2. **A coding agent** doing JIT lookup during implementation — needs structured, citable, machine-readable reference cards

The source curriculum is `distributed_video_dit_inference.md` — 24 topics, ~85 resources.

## Repository structure

```
distributed_video_dit_inference.md   # source curriculum (read-only reference)
sources/
  <id>/
    raw/                             # Tier 1: original fetched artifacts
      page.html                      #   blogs/docs: saved HTML
      paper.pdf                      #   arXiv: downloaded PDF
      issue.json                     #   GitHub: API response
    full.md                          # Tier 2: lossless markdown conversion
refs/
  manifest.yaml                      # master inventory of all resources
  resources/<id>.md                  # Tier 3: condensed resource cards
  topics/<nn>-<slug>.md              # per-topic synthesis docs
```

## Three-tier extraction pipeline

Each resource goes through three tiers. These are **separate operations** with different fidelity requirements and can be run by different models or tools.

### Tier 1 — Raw fetch (`sources/<id>/raw/`)

Store the original artifact exactly as retrieved. This is the ground truth — everything else derives from it.

- **Blogs/docs**: Save HTML with `curl` or equivalent. Filename: `page.html`
- **arXiv papers**: Download PDF from `https://arxiv.org/pdf/<arxiv-id>`. Filename: `paper.pdf`. Also save the abstract page HTML.
- **GitHub issues/RFCs**: Save via `gh api`. Filename: `issue.json`
- **GitHub source files**: Save the raw file content. Filename: original filename.
- **Project pages**: Save HTML. Filename: `page.html`

This tier is mechanical — no LLM needed, just fetch tools.

**Manifest status after**: `pending` → `fetched`

### Tier 2 — Lossless markdown conversion (`sources/<id>/full.md`)

Convert the raw artifact to clean, complete markdown. **No summarization, no editorial judgment.** The full text, all sections, all content — just in a readable, searchable, agent-friendly format.

- YAML frontmatter: `title`, `source_url`, `fetch_date`, `source_type`, `author` (if applicable)
- Preserve ALL section headings as markdown headings
- Preserve all code blocks, formulas, API signatures verbatim
- Preserve all tables
- Describe figures/diagrams in brackets: `[Figure: description of what the figure shows]`
- For papers: preserve abstract, all sections, all equations, all tables, references
- For docs: preserve all sections including API details, parameters, examples
- For GitHub issues: preserve the original post body and key follow-up comments

**Tooling preference**: Use the highest-fidelity conversion available:
- HTML → markdown: pandoc, or a strong model reading the HTML
- PDF → markdown: marker, nougat, or a strong model reading the PDF
- JSON (GitHub) → markdown: direct formatting from structured data

Aim for **zero information loss** relative to the raw artifact. If in doubt, include more rather than less. This file is the searchable ground truth that Tier 3 citations point back to.

**Manifest status after**: `fetched` → `converted`

### Tier 3 — Smart condensation (`refs/resources/<id>.md`)

This is the resource card — the agent-facing reference. **This tier requires reading comprehension and project-specific judgment.** Use a strong model.

Each card has a **type-appropriate template** (already stubbed). Required fields for all types:

- **Source** (URL), **Type**, **Topics**, **Status**
- **Why it matters** — 1-2 sentences on relevance to the project (pre-populated from source doc)
- **Core claims** — each with `**Evidence**: sources/<id>/full.md#<heading>` citation
- **Actionables / gotchas** — implementation implications for distributed video DiT inference
- **Related resources** — cross-references to other resource IDs

Optional sections vary by type:
- Papers: "Key technical details" (formulas, algorithms, architecture)
- Docs: "Key sections", "API surface / configuration"
- Blogs: "Key insights"
- Code/RFCs: "Problem statement", "Design decisions", "Key APIs / interfaces"

**Manifest status after**: `converted` → `condensed`

## Reference implementation

`making-dl-go-brrrr` is the completed reference (currently has Tier 2 + Tier 3 but needs Tier 1 raw fetch backfilled). See:
- `sources/making-dl-go-brrrr/extracted.md` → rename to `full.md` (this is the Tier 2 output)
- `refs/resources/making-dl-go-brrrr.md` (this is the Tier 3 output)

## Manifest (`refs/manifest.yaml`)

Single source of truth for all resources. Fields: `id`, `title`, `urls`, `type`, `topics`, `priority`, `status`, `local_paths`, `notes`.

Status values: `pending` → `fetched` → `converted` → `condensed` | `link_only` (hard-to-fetch).

Priority: `high` (Phase 1, ~15 resources), `medium`, `low`.

## Phasing

- **Phase 1**: The 15 `priority: high` resources — do these first, following the suggested reading order in the source doc
- **Phase 2**: Backfill by topic as implementation pressure hits
- **`link_only` resources**: Google Docs, YouTube, paywalled ACM, missing CVPR URL — skip unless they become load-bearing

## Quality expectations

- **Tier 2 (full.md)**: Zero information loss. If a sentence is in the original, it's in full.md. No editorial cuts.
- **Tier 3 (resource cards)**: Every non-trivial claim must cite a specific heading or section in `sources/<id>/full.md`
- Don't invent claims — if the source doesn't support it, don't include it
- Flag uncertainty with `(unverified)` rather than guessing at hardware specs, numeric ranges, or protocol details
- Preserve the author's terminology and framing — don't silently reinterpret
- Keep cards focused on what matters for **distributed video DiT inference**, not general ML knowledge

## What NOT to do

- Don't modify `distributed_video_dit_inference.md` (source curriculum) unless fixing a known error
- Don't create resource cards for non-Phase-1 resources yet (stubs exist only for Phase 1)
- Don't summarize full.md — it must be the complete source text
- Don't add speculative "future work" sections to cards
- Don't fetch `link_only` resources without being explicitly asked
- Don't collapse Tier 1 and Tier 2 into one step — keep the raw artifact separate from the markdown conversion
