# Distributed Video DiT Inference — Learning Reference Library

## Project purpose

This repo is a **grounded reference library** for building a pipeline-parallel inference system for a video Diffusion Transformer. It serves two consumers:

1. **A human learner** studying the material topic-by-topic
2. **A coding agent** doing JIT lookup during implementation — needs structured, citable, machine-readable reference cards

The source curriculum is `distributed_video_dit_inference.md` — 24 topics, ~85 resources.

## Repository structure

```
distributed_video_dit_inference.md   # source curriculum (read-only reference)
refs/
  manifest.yaml                      # master inventory of all resources
  resources/<id>.md                  # per-resource reference cards
  topics/<nn>-<slug>.md              # per-topic synthesis docs
sources/
  <id>/                              # fetched source material per resource
    extracted.md                     # clean full-text markdown extraction
```

## Manifest (`refs/manifest.yaml`)

Single source of truth for all resources. Fields: `id`, `title`, `urls`, `type`, `topics`, `priority`, `status`, `local_paths`, `notes`.

Status values: `pending` → `fetched` → `extracted` → `summarized` | `link_only` (hard-to-fetch).

Priority: `high` (Phase 1, ~15 resources), `medium`, `low`.

## How to extract a resource

Follow this two-file pattern (see `making-dl-go-brrrr` as the reference implementation):

### 1. Fetch and extract: `sources/<id>/extracted.md`

- YAML frontmatter: `title`, `source_url`, `fetch_date`, `source_type`, `author` (if applicable)
- Full text in markdown — do NOT summarize or truncate
- Preserve all section headings, code blocks, formulas, API signatures
- Describe figures/diagrams in brackets: `[Figure: description]`
- For arXiv papers: fetch abstract page + PDF content
- For docs: capture all relevant sections
- For GitHub issues/RFCs: capture the original post and key design discussion

### 2. Update resource card: `refs/resources/<id>.md`

Each card has a **type-appropriate template** (already stubbed). Required fields for all types:

- **Source** (URL), **Type**, **Topics**, **Status**
- **Why it matters** — 1-2 sentences on relevance to the project (pre-populated from source doc)
- **Core claims** — each with `**Evidence**: sources/<id>/extracted.md#<heading>` citation
- **Actionables / gotchas** — implementation implications for distributed video DiT inference
- **Related resources** — cross-references to other resource IDs

Optional sections vary by type (papers get "Key technical details"; docs get "API surface / configuration"; blogs get "Key insights"; code/RFCs get "Design decisions").

Update Status from `stub` to `extracted` when complete.

### 3. Update manifest

Change `status: pending` → `status: extracted` in `refs/manifest.yaml` for the completed resource.

## Phasing

- **Phase 1**: The 15 `priority: high` resources — do these first, following the suggested reading order in the source doc
- **Phase 2**: Backfill by topic as implementation pressure hits
- **`link_only` resources**: Google Docs, YouTube, paywalled ACM, missing CVPR URL — skip unless they become load-bearing

## Quality expectations

- Every non-trivial claim in a resource card must cite a specific heading or section in `sources/<id>/extracted.md`
- Don't invent claims — if the source doesn't support it, don't include it
- Flag uncertainty with `(unverified)` rather than guessing at hardware specs, numeric ranges, or protocol details
- Preserve the author's terminology and framing — don't silently reinterpret
- Keep cards focused on what matters for **distributed video DiT inference**, not general ML knowledge

## What NOT to do

- Don't modify `distributed_video_dit_inference.md` (source curriculum) unless fixing a known error
- Don't create resource cards for non-Phase-1 resources yet (stubs exist only for Phase 1)
- Don't summarize extracted.md — it should be the full source text
- Don't add speculative "future work" sections to cards
- Don't fetch `link_only` resources without being explicitly asked
