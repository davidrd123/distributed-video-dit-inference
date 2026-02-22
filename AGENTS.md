# Distributed Video DiT Inference — Learning Reference Library

## Project purpose

This repo is a **grounded reference library** for building a pipeline-parallel inference system for a video Diffusion Transformer. It serves two consumers:

1. **A human learner** studying the material topic-by-topic
2. **A coding agent** doing JIT lookup during implementation — needs structured, citable, machine-readable reference cards

The source curriculum is `distributed_video_dit_inference.md` — 24 topics, ~85 resources.

## Sibling repo: `scope-drd`

This reference library is designed to interop with `scope-drd/` (the implementation repo for the PP inference system). They live as **sibling directories** under a shared parent — no submodules, no nesting.

- **This repo → scope-drd**: `refs/implementation-context.md` cites scope-drd working notes paths (e.g., `scope-drd/notes/FA4/h200/tp/feasibility.md`) to ground resource cards in measured findings.
- **scope-drd → this repo**: Implementation notes can cite resource cards by ID (e.g., `refs/resources/zero-bubble-pp.md` claim 3) for JIT lookup during development.
- **The human is the bridge** — neither repo depends on the other at build time. Cross-references are documentation pointers, not imports.
- **Multi-machine**: Both repos are cloned independently on each machine (desktop, laptop, remote server). The sibling layout is a convention, not enforced — absolute paths will differ. All cross-references use **relative paths from the repo root** (e.g., `scope-drd/notes/...` means "the scope-drd sibling, then that path within it"), never absolute paths. The sibling directory name may vary (`scope-drd/` locally, `scope/` on remote) — references here use `scope-drd/` canonically.

This convention may evolve as the interop pattern matures.

## Repository structure

```
distributed_video_dit_inference.md   # source curriculum (read-only reference)
sources/
  <id>/
    raw/                             # Tier 1: original fetched artifacts
      page.html                      #   blogs/docs: saved HTML
      paper.pdf                      #   arXiv: downloaded PDF
      abstract.html                  #   arXiv: abstract page
      issue.json                     #   GitHub issues: API response
      repo/                          #   GitHub repos: cloned source (key files only)
    full.md                          # Tier 2: lossless markdown conversion
refs/
  manifest.yaml                      # master inventory of all resources
  resources/<id>.md                  # Tier 3: condensed resource cards
  topics/<nn>-<slug>.md              # per-topic synthesis docs
  implementation-context.md          # bridge: working notes findings → load-bearing resources
```

## Three-tier extraction pipeline

Each resource goes through three tiers. These are **separate operations** with different fidelity requirements and can be run by different models or tools.

### Tier 1 — Raw fetch (`sources/<id>/raw/`)

Store the original artifact exactly as retrieved. This is the ground truth — everything else derives from it.

- **Blogs/docs**: Save HTML with `curl` or equivalent. Filename: `page.html`
- **arXiv papers**: Download PDF from `https://arxiv.org/pdf/<arxiv-id>`. Filename: `paper.pdf`. Also save the abstract page HTML as `abstract.html`.
- **GitHub issues/RFCs**: Save via `gh api`. Filename: `issue.json`
- **GitHub repos** (code resources): Clone or fetch key source files into `raw/repo/`. Don't clone the full history — fetch specific files/directories that are architecturally relevant. Include README, core source modules, config files. Skip tests, CI, docs-only files unless specifically relevant.
- **Project pages**: Save HTML. Filename: `page.html`

This tier is mechanical — no LLM needed, just fetch tools.

**Manifest status after**: `pending` → `fetched`

### Tier 2 — Lossless markdown conversion (`sources/<id>/full.md`)

Convert the raw artifact to clean, complete markdown. **No summarization, no editorial judgment.** The full text, all sections, all content — just in a readable, searchable, agent-friendly format.

- YAML frontmatter: `title`, `source_url`, `fetch_date`, `source_type`, `author` (if applicable), `conversion_notes` (for papers — document any equation/table conversion artifacts)
- Preserve ALL section headings as markdown headings
- Preserve all code blocks, formulas, API signatures verbatim
- Preserve all tables
- Describe figures/diagrams in brackets: `[Figure: description of what the figure shows]`
- For papers: preserve abstract, all sections, all equations, all tables, references
- For HTML docs/blogs: strip site chrome (nav, footer, sidebar) only. Keep **all** article body content verbatim — no selectivity, no "include relevant sections." Consistent policy across all docs resources.
- For docs: preserve all sections including API details, parameters, examples
- For GitHub issues: preserve the original post body and key follow-up comments
- For GitHub repos: `full.md` is a **structured code dump** — file tree followed by verbatim file content under `## File: path/to/file.py` headings. **No interpretation, no architecture overview** — that's Tier 3 work. The `full.md` is a structured, citable source that Tier 3 references.

**Tooling preference**: Use the highest-fidelity conversion available:
- HTML → markdown: pandoc, or a strong model reading the HTML
- PDF → markdown (arXiv papers): **Try LaTeX source first**, fall back to vision.
  - **Recipe A — LaTeX source (preferred for arXiv papers)**:
    1. Fetch source bundle: `curl -sL -o /tmp/<id>_src.tar.gz "https://arxiv.org/e-print/<arxiv-id>"`
    2. Extract: `mkdir -p /tmp/<id>_src && tar xzf /tmp/<id>_src.tar.gz -C /tmp/<id>_src`
    3. Find the main `.tex` file and `.bbl` (bibliography). Read these directly — equations, tables, and section structure are exact ground truth from the authors.
    4. Convert LaTeX markup to markdown: `\section{}` → `##`, `\begin{equation}` → `$$...$$`, `\begin{tabular}` → markdown tables, `\cite{key}` → `[N]` references.
    5. For figures: describe in brackets using the `\caption{}` text from the `.tex` + the figure filename for context. The actual images are in the source bundle but don't commit them.
    6. For bibliography: convert `.bbl` entries to a numbered reference list.
    7. Add `conversion_notes: "Converted from LaTeX source via arxiv e-print"` to YAML frontmatter.
    8. Clean up `/tmp/<id>_src/` after conversion.
	  - **Recipe B — Vision fallback (non-arXiv PDFs, or when e-print is unavailable)**:
	    1. Render pages to PNGs (do **not** commit these): `mkdir -p /tmp/<id>_pages && pdftoppm -png -r 200 sources/<id>/raw/paper.pdf /tmp/<id>_pages/page`
	    2. Read the rendered pages directly as images (multimodal input), reconstructing section structure, equations, tables, and bracketed figure descriptions.
	    3. Cross-check completeness with text extraction (especially for References and long tables):
	       - Prefer `-nopgbrk` to avoid form-feed artifacts: `pdftotext -raw -nopgbrk sources/<id>/raw/paper.pdf /tmp/<id>_raw.txt`
	       - Use `-layout` when reconstructing tables: `pdftotext -layout sources/<id>/raw/paper.pdf /tmp/<id>_layout.txt`
	       - Use `-f/-l` to isolate a single page when an equation/table is tricky: `pdftotext -raw -f 7 -l 7 sources/<id>/raw/paper.pdf /tmp/<id>_p7.txt`
	    4. Add `conversion_notes:` to YAML frontmatter documenting the method + any equation/table artifacts.
	    5. Clean up `/tmp/<id>_pages/` after conversion.
	  - Alternative tools (lower quality but faster): marker, nougat
- JSON (GitHub) → markdown: direct formatting from structured data
- Repo → structured code dump: file tree + verbatim content under `## File:` headings (no interpretation — that's Tier 3)

Aim for **zero information loss** relative to the raw artifact. If in doubt, include more rather than less. This file is the searchable ground truth that Tier 3 citations point back to.

**Manifest status after**: `fetched` → `converted`

### Tier 3 — Smart condensation (`refs/resources/<id>.md`)

This is the resource card — the agent-facing reference. **This tier requires reading comprehension and project-specific judgment.** Any agent can write Tier 3 cards when asked. Reference both `sources/<id>/full.md` and `refs/implementation-context.md` for grounding. **Do not update manifest status to `condensed`** — that happens after user review.

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
- **Repos**: "Architecture overview", "Key modules" (with file paths into `raw/repo/`), "Design patterns", "Configuration / entry points"

**Manifest status after**: `converted` → `condensed`

## Reference implementations

Two completed references exist covering the main resource types:

### Blog/HTML: `making-dl-go-brrrr`
- Tier 1: `sources/making-dl-go-brrrr/raw/page.html` (159KB)
- Tier 2: `sources/making-dl-go-brrrr/full.md` (~3700 words)
- Tier 3: `refs/resources/making-dl-go-brrrr.md` (7 claims, 6 insights, 8 actionables)

### Paper/PDF: `dit-paper`
- Tier 1: `sources/dit-paper/raw/paper.pdf` (43MB) + `raw/abstract.html`
- Tier 2: `sources/dit-paper/full.md` (~8500 words, all tables/equations/63 references)
- Tier 3: `refs/resources/dit-paper.md` (7 claims, full architecture spec with configs, 8 actionables)

## Manifest (`refs/manifest.yaml`)

Single source of truth for all resources. Fields: `id`, `title`, `urls`, `type`, `topics`, `priority`, `status`, `local_paths`, `notes`.

Status values: `pending` → `fetched` → `converted` → `condensed` | `link_only` (hard-to-fetch).

Priority: `high` (Phase 1, ~15 resources), `medium`, `low`.

## Phasing

- **Phase 1**: The 15 `priority: high` resources — do these first, following the suggested reading order in the source doc
- **Phase 2**: Backfill by topic as implementation pressure hits
- **`link_only` resources**: Google Docs, YouTube, paywalled ACM, missing CVPR URL — skip unless they become load-bearing

## Process observations (from first two extractions)

These notes capture what we learned doing the `making-dl-go-brrrr` and `dit-paper` extractions. Use them to avoid repeating mistakes.

### Tier 2 "lossless" is aspirational for some sources

- **WebFetch summarizes by default.** You cannot get verbatim full text through it for long content. For true lossless Tier 2, derive from the Tier 1 raw artifact (HTML or PDF), not from a web fetch.
- **PDF extraction via a strong model reading rendered pages** is currently the highest-quality approach for papers. Read the PDF page-by-page, reconstruct section structure, tables, and equations. This is expensive but faithful.
- **Blog posts lose ~20-25%** even with careful extraction (the `making-dl-go-brrrr` full.md is ~3700 words from a ~4500-5000 word original). Acceptable for blogs, but for papers and docs aim for genuinely complete text.
- **Figures can't be reproduced in markdown.** Describe them in brackets with enough detail that an agent can understand what the figure shows without seeing it. Include axis labels, data series, and key takeaways.

### Tier 3 quality depends on understanding the project context

- The "Actionables / gotchas" section is the most valuable part of a resource card for agent JIT lookup. It answers "so what does this mean for *our* pipeline-parallel video DiT system?" Generic summaries are low value.
- **Cross-reference related resources** — an agent looking at the DiT paper card should be pointed to the PP scheduling papers and the performance analysis resources. These links are what make the library navigable.
- **Architecture details matter for papers.** For the DiT paper, capturing the exact model configs (layers, hidden dim, heads, Gflops) and the adaLN-Zero conditioning mechanism was essential — an agent partitioning the model across pipeline stages needs these specifics.

### Practical extraction workflow

1. **Tier 1 first, always.** Fetch raw artifacts before doing anything else. `curl` for HTML, `curl` for PDFs, `gh api` for GitHub issues. This is fast and gives you the ground truth to work from.
2. **Tier 2 from Tier 1, not from web fetch.** Read the raw artifact directly (HTML file, PDF pages) rather than re-fetching through a summarizing tool.
3. **Tier 3 from Tier 2.** Read `full.md` to write the resource card. This ensures your citations actually point to content that exists in `full.md`.
4. **Update manifest last.** Set status and `local_paths` after all tiers are complete for that resource.

### Resource types not yet templated

- **GitHub repos** (code): StreamDiffusionV2 is the first repo-type resource. The pattern is: clone/fetch key files into `raw/repo/`, write Tier 2 as file tree + verbatim code under `## File:` headings (no interpretation), condense into Tier 3 resource card focusing on architecture, key APIs, and design patterns relevant to the project. Interpretation and architecture overview is Tier 3 only.
- **Large docs pages**: PyTorch CUDA Semantics, NCCL User Guide — these are sprawling multi-section docs. Strip site chrome only; keep all article body content verbatim.

## Quality expectations

- **Tier 2 (full.md)**: Zero information loss for papers and blog posts. For HTML docs, strip site chrome only — keep all article body content verbatim. For repos, file tree + verbatim code under `## File:` headings.
- **Tier 3 (resource cards)**: Every non-trivial claim must cite a specific heading or section in `sources/<id>/full.md`
- Don't invent claims — if the source doesn't support it, don't include it
- Flag uncertainty with `(unverified)` rather than guessing at hardware specs, numeric ranges, or protocol details
- Preserve the author's terminology and framing — don't silently reinterpret
- Keep cards focused on what matters for **distributed video DiT inference**, not general ML knowledge

## What NOT to do

- Don't modify `distributed_video_dit_inference.md` (source curriculum) unless fixing a known error
- Don't create resource cards for non-Phase-1 resources yet (stubs exist only for Phase 1)
- Don't summarize full.md — it must be the complete source text (or explicitly annotated subset for large docs/repos)
- Don't add speculative "future work" sections to cards
- Don't fetch `link_only` resources without being explicitly asked
- Don't collapse Tier 1 and Tier 2 into one step — keep the raw artifact separate from the markdown conversion
- Don't clone full git history for repo resources — fetch specific files/directories

## Contributing to this document

This file is a living document. If you discover techniques, flags, or workflows that improve extraction quality, **add them here** (e.g., refining the PDF recipe, noting a tool that works well for a specific source type). Keep additions targeted and note what you tested. This builds institutional memory across agents and sessions.
