# Reference Library Completion — Task Flow & Ordering

Date: 2026-02-22
Last updated: 2026-02-22 (opus1 — Batch B1 complete, cost lesson learned)

---

## Agent coordination

Multiple agents work on this library. Each owns specific tracks to avoid duplication and wasted budget.

| Agent | Identity | Strengths | Assigned work |
|---|---|---|---|
| **opus** | Claude Opus (Claude Code) | PDF reading, project-specific judgment, Tier 3 grounding, QC | Batch B2 (papers), Track C (Tier 3 cards), review |
| **gpt-xhigh** | GPT 5.2 xhigh (Codex CLI) | Thorough, cost-effective, good for bulk work | Track A, mechanical Tier 2 (HTML/repo/JSON), bulk file ops, consistency scripts |

GPT 5.2 xhigh via Codex CLI is the workhorse — more usage headroom on OpenAI plan. Opus is reserved for high-judgment work only.

### Cost discipline (learned 2026-02-22)

Opus burned **26% of 5-hour budget in 12 minutes** doing Batch B1 (HTML→markdown) — work that was mostly pandoc + cleanup. This is expensive brainpower for text munging.

**Rules going forward:**
- **Opus**: Only for work that requires strong reading comprehension — PDF page reading (equations, tables, figures), Tier 3 cards (project-specific judgment, working notes integration), topic synthesis
- **Codex / GPT xhigh**: All mechanical Tier 2 conversions (HTML, JSON, repo dumps), file tree operations, manifest updates, context expansion wiring
- **Before launching Opus subagents**: Ask "does this need reading comprehension or just text processing?" If the latter, don't use Opus.

### Handoff conventions

- Each agent should note its identity (e.g., `opus1`, `codex1`) in commit messages and file headers where relevant
- PLAN.md is the coordination doc — check it before starting a new track
- Mark work in progress in the "Progress log" section below to prevent overlap
- Manifest is the source of truth for resource status

## Progress log

| Date | Agent | Work done | Notes |
|---|---|---|---|
| 2026-02-22 | opus1 | CLAUDE.md policy updates | Tier 2 policies codified |
| 2026-02-22 | opus1 | Batch B1 complete (7 resources → Tier 2) | funcol-rfc, ezyang, dynamo, nccl, pytorch-cuda-semantics, pipelining-api, cuda-graphs. NCCL required sub-page fetch. |
| 2026-02-22 | opus2 | QC of medium-priority Tier 1 artifacts | Fixed ezyang-blog URL, fetched fp-non-assoc PDF, deleted dupes, fetched issue comments. Committed `bae52f5`. |
| 2026-02-22 | opus2 | PLAN.md moved from scope-drd, sibling repo convention | Committed `8136655`. |
| 2026-02-22 | codex2+opus2 | Batch B3 complete (streamdiffusionv2 → Tier 2) | 42 files, 9597 lines structured dump. Committed `fe92551`. |
| | | | |

---

## Context

The reference library has 15 Phase 1 (high-priority) resources. 2 are complete (Tier 3). 13 have raw artifacts (Tier 1) but need Tier 2 (lossless markdown) and Tier 3 (condensed resource card with implementation-grounded actionables). There's also the implementation context expansion work (bridge doc + topic stubs). The goal is to get the library to a state where an agent can do JIT lookup during PP bringup.

## Current state

| Resource | Type | Raw size | Tier | Notes |
|---|---|---|---|---|
| `making-dl-go-brrrr` | blog | 159KB HTML | **3 (done)** | Reference implementation |
| `dit-paper` | paper | 43MB PDF | **3 (done)** | Reference implementation |
| `funcol-rfc-93173` | RFC (GitHub) | 11KB JSON | **2** | opus1, 2026-02-22 |
| `ezyang-state-of-compile` | blog | 26KB HTML | **2** | opus1, 2026-02-22 |
| `dynamo-deep-dive` | docs | 208KB HTML | **2** | opus1, 2026-02-22 |
| `nccl-user-guide` | docs | 107KB+590KB (multi-page) | **2** | opus1, 2026-02-22. Tier 1 expanded with 7 sub-pages |
| `pytorch-cuda-semantics` | docs | 378KB HTML | **2** | opus1, 2026-02-22 |
| `cuda-graphs-guide` | docs | 312KB HTML | **2** | opus1, 2026-02-22 |
| `pytorch-pipelining-api` | docs | 582KB HTML | **2** | opus1, 2026-02-22 |
| `streamdiffusionv2` | repo+paper | 42KB HTML + repo (42 files) | **2** | codex2+opus2, 2026-02-22. Structured code dump (9597 lines, streamv2v/ first) |
| `gpipe` | paper | 539KB PDF | 1 | |
| `pipedream-2bw` | paper | 2.2MB PDF | 1 | |
| `zero-bubble-pp` | paper | 649KB PDF | 1 | |
| `pipedit` | paper | 3.9MB PDF | 1 | |
| `pagedattention` | paper | 1.5MB PDF | 1 | |

## Tier 2 policies (updated per Codex review)

These override the original CLAUDE.md definitions where they conflict. Update CLAUDE.md to match.

- **HTML docs/blogs**: Strip site chrome (nav, footer, sidebar) only. Keep **all** article body content verbatim. No selectivity — no "include relevant sections." Consistent policy across all docs resources.
- **Papers (PDF)**: Opus reads PDF pages directly (page by page). Add `conversion_notes:` field to YAML frontmatter documenting any equation/table conversion artifacts. This is the highest-quality approach but expensive.
- **GitHub repos**: Tier 2 = file tree + verbatim content under `## File: path/to/file.py` headings. **No interpretation, no architecture overview** — that's Tier 3. The `full.md` is a structured dump that Tier 3 cites. Update CLAUDE.md repo Tier 2 definition accordingly.
- **GitHub issues/RFCs**: Direct formatting from JSON structured data.

## Task flow (4 tracks, partially parallel)

### Track A: Implementation context expansion (Codex)
**Dependencies**: None (can start immediately)
**Effort**: ~1 Codex session
**What**: Add implementation context sections to remaining ~17 topic stubs, expand bridge doc with medium-priority resources. **Scope guard**: wiring, pointers, and TODOs only — do NOT rewrite actionables in completed cards (that's Track C work).

### Track B: Tier 2 extraction (batched by type)
**Dependencies**: None (can start immediately, parallel with Track A)
**Effort**: ~3-4 sessions

Tier 2 is lossless conversion (raw → markdown, no summarization, no interpretation).

**Batch B1 — Small/easy + infra docs (1-2 sessions, Codex):**
1. `funcol-rfc-93173` — 11KB JSON → markdown. Smallest, fastest.
2. `ezyang-state-of-compile` — 26KB HTML → markdown. Short blog post.
3. `dynamo-deep-dive` — 208KB HTML → markdown. Medium blog/tutorial.
4. `nccl-user-guide` — 107KB HTML → markdown. Full article body, strip nav only.
5. `pytorch-cuda-semantics` — 378KB HTML → markdown. Full article body, strip nav only.
6. `pytorch-pipelining-api` — 582KB HTML → markdown. Full article body, strip nav only.
7. `cuda-graphs-guide` — 312KB HTML → markdown. Full article body, strip nav only.

Infra docs pulled forward (was Week 3, now Week 1) because they're the "why did this hang / what does this env var do" reference during PP bringup.

**Batch B2 — Papers (1-2 sessions, Opus reads PDF pages):**
8. `gpipe` — 539KB PDF. Short, foundational.
9. `pagedattention` — 1.5MB PDF. Moderate length.
10. `zero-bubble-pp` — 649KB PDF. Focused.
11. `pipedream-2bw` — 2.2MB PDF. Medium-length.
12. `pipedit` — 3.9MB PDF. Longer paper.

Each paper gets `conversion_notes:` in frontmatter noting any equation/table artifacts.

**Batch B3 — Repo (1 session, Codex):**
13. `streamdiffusionv2` — 42 source files already fetched in `raw/repo/`. Tier 2 = file tree + verbatim code under `## File:` headings. No interpretation. Prioritize `streamv2v/` files (the distributed inference code) over `causvid/` files.

### Track C: Tier 3 condensation (Opus, user reviews each card)
**Dependencies**: Each resource's Tier 2 must be complete first
**Effort**: ~3-4 sessions (sequential, gated by user review)

Tier 3 requires project-specific judgment — reading `full.md` and writing actionables grounded in the working notes. User reviews every card before marking `condensed`.

Order by PP bringup urgency (from `implementation-context.md`):

**Wave C1 — PP-critical (first):**
1. `streamdiffusionv2` — most load-bearing for PP bringup
2. `gpipe` — foundational PP concept
3. `funcol-rfc-93173` — compile-aware collectives for mesh_pg

**Wave C2 — Compile/Dynamo + infra (second):**
4. `dynamo-deep-dive` — graph breaks, SymInt, guards
5. `ezyang-state-of-compile` — compile + distributed story
6. `nccl-user-guide` — algorithm selection, env vars, debugging
7. `pytorch-cuda-semantics` — stream ordering for collectives

**Wave C3 — PP infrastructure (third):**
8. `pytorch-pipelining-api` — PP schedule API
9. `pagedattention` — KV-cache lifecycle patterns
10. `cuda-graphs-guide` — CUDAGraph capture (v2.0 roadmap)

**Wave C4 — Scheduling theory (fourth):**
11. `pipedream-2bw` — 1F1B schedule
12. `zero-bubble-pp` — advanced PP scheduling
13. `pipedit` — pipelined sequence parallelism

Also in Wave C1: sharpen actionables in the 2 already-completed cards (`making-dl-go-brrrr`, `dit-paper`) using `implementation-context.md` guide table.

### Track D: Topic synthesis docs (last, after C)
**Dependencies**: Tier 3 cards for the topic's resources must exist
**Effort**: ~1-2 sessions, can be incremental

Fill in the synthesis sections of `refs/topics/*.md` — mental model, key concepts, cross-resource agreement/disagreement, practical checklist, experiments to run. This is the highest-judgment work and only makes sense once the underlying resource cards are populated. PP-critical topics first: 13, 11, 16, 24, 22.

## Execution order

```
Week 1:
  Track A (Codex): implementation context expansion      ─── can start now
  Batch B1 (Codex): small/easy + infra docs (7 resources) ─ can start now, parallel
  Batch B3 (Codex): streamdiffusionv2 repo structured dump ─ can start now, parallel

Week 2:
  Batch B2 (Opus): 5 papers via PDF page reading
  Wave C1 (Opus + user review): condense streamdiffusionv2, gpipe, funcol-rfc
    + sharpen making-dl-go-brrrr + dit-paper actionables

Week 3:
  Wave C2 (Opus + user review): condense dynamo, ezyang, nccl-user-guide, pytorch-cuda-semantics

Week 4:
  Wave C3+C4 (Opus + user review): condense remaining 6 cards
  Track D: begin topic synthesis for PP-critical topics (13, 11, 16, 24, 22)
```

## Assignment guidance

| Track | Agent | Why |
|---|---|---|
| Track A (context expansion) | **gpt-xhigh** | Reading + targeted edits across many files; no deep judgment |
| Track B, B1 (HTML Tier 2) | **DONE** | Completed by opus (2026-02-22). |
| Track B, B3 (repo Tier 2) | **DONE** | Completed by gpt-xhigh+opus (2026-02-22). |
| Track B, B2 (Paper Tier 2) | **opus** | PDF page reading requires strong model for equation/table fidelity |
| Track C (Tier 3) | **opus**, user reviews each card | Requires project-specific judgment, working notes integration |
| Track D (topic synthesis) | User + **opus** | Highest-judgment work; user's understanding matters most |

### Track C review workflow
Opus writes each Tier 3 card, presents it to the user for review. User can:
- Approve → mark `condensed` in manifest
- Request changes → Opus revises
- Add context → user provides working notes details Opus missed

This keeps cards grounded but doesn't block Tier 2 extraction (which runs ahead).

## CLAUDE.md updates needed (prerequisite for Track B) — DONE (opus1, 2026-02-22)

1. ~~Repo Tier 2 definition: change "annotated source guide" to "file tree + verbatim content under `## File:` headings"~~ ✓
2. ~~HTML docs Tier 2 policy: add "strip site chrome only, keep all article body verbatim"~~ ✓
3. ~~Paper Tier 2: add `conversion_notes:` frontmatter requirement~~ ✓
4. ~~Add `implementation-context.md` to repo structure diagram~~ ✓ (already done)

## Codex session prompts

Each Codex session should get:
1. `CLAUDE.md` (repo structure + extraction pipeline docs)
2. The specific batch of resources to process
3. Reference implementations to follow (`making-dl-go-brrrr` for blog/HTML, `dit-paper` for paper/PDF)
4. For Track C: `refs/implementation-context.md` (the per-card actionables sharpening guide)
5. For Track B (papers): instruction to add `conversion_notes:` to frontmatter

## Verification

After each batch:
- `refs/manifest.yaml` status updated (`fetched` → `converted` for Tier 2, `converted` → `condensed` for Tier 3)
- `sources/<id>/full.md` exists and has YAML frontmatter (including `conversion_notes:` for papers)
- For repos: `full.md` has `## File:` headings matching the file tree
- For Tier 3: `refs/resources/<id>.md` has filled claims with `Evidence:` citations pointing to `full.md` headings
- Tier 3 actionables reference specific working notes findings (not generic statements)

## What NOT to do yet

- Phase 2 resources (medium/low priority) — only backfill when implementation pressure hits
- Topic synthesis for non-PP-critical topics — defer until the PP-critical topics are done
- `link_only` resources — skip unless they become load-bearing
