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
- **Opus**: Tier 3 cards (project-specific judgment, working notes integration), topic synthesis, review. **Not** PDF Tier 2 — too expensive (see below).
- **Codex / GPT 5.2 xhigh**: All Tier 2 conversions including PDFs (extract images from PDF, read directly), HTML, JSON, repo dumps, file tree operations, manifest updates, context expansion wiring
- **Before launching Opus subagents**: Ask "does this need reading comprehension or just text processing?" If the latter, don't use Opus.

### Cost discipline update (learned 2026-02-22, session 2)

Opus PDF page reading burned **45% of 5-hour budget in 40 minutes** doing B2 (3 papers: gpipe, zero-bubble-pp, pipedream-2bw). The rendered-page-image approach is high quality but extremely expensive per token. GPT 5.2 xhigh should try extracting images from PDFs and reading directly — potentially comparable quality at much lower Opus cost. Remaining B2 paper (`pipedit`) assigned to GPT 5.2.

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
| 2026-02-22 | opus2 | pagedattention → Tier 2 | 16-page PDF read page-by-page, full lossless conversion with all equations/tables/figures/references. |
| 2026-02-22 | codex1 | NCCL user guide → Tier 2 rebuild | Reassembled `sources/nccl-user-guide/full.md` to cover setup/communicators/thread-safety/API/troubleshooting; `py-modindex.html` link was 404 at fetch time. |
| 2026-02-22 | opus1 | B2: gpipe, zero-bubble-pp, pipedream-2bw → Tier 2 | 3 papers via PDF page reading. 45% of 5hr quota in 40min — reassigning pipedit to GPT 5.2. |
| 2026-02-22 | gpt-xhigh | pipedit → Tier 2 | 10pp + supplementary, dense tables preserved as monospaced blocks. Spot-checked by opus2 — numbers match PDF. |
| 2026-02-22 | — | **Batch B2 complete, all Phase 1 Tier 2 done** | 15/15 Phase 1 resources at Tier 2+. Next: Track C (Tier 3 condensation). |
| 2026-02-22 | opus3 | Created `refs/reading-guide.md` | Bringup-synced study guide (spine + phases + topic index). Derived from GPT-5.2 integrated learning path, reconciled with canonical catalog. |
| 2026-02-22 | opus3 | CausVid fetched (Tier 1), manifest updated | 14.6MB PDF + abstract. Added as Phase 1 (topics 22, 24). Also added SDv2 arxiv URL to manifest. |
| | | | |

---

## Context

The reference library has 15 Phase 1 (high-priority) resources. 2 are complete (Tier 3). 13 have raw artifacts (Tier 1) but need Tier 2 (lossless markdown) and Tier 3 (condensed resource card with implementation-grounded actionables). There's also the implementation context expansion work (bridge doc + topic stubs). The goal is to get the library to a state where an agent can do JIT lookup during PP bringup.

## Current state

| Resource | Type | Raw size | Tier | Notes |
|---|---|---|---|---|
| `making-dl-go-brrrr` | blog | 159KB HTML | **3 (done)** | Reference implementation |
| `dit-paper` | paper | 43MB PDF | **3 (done)** | Reference implementation |
| `funcol-rfc-93173` | RFC (GitHub) | 11KB JSON | **2** | opus1, 2026-02-22. TODO: Tier 2 is missing follow-up comments (issue.json reports 44 comments). |
| `ezyang-state-of-compile` | blog | 26KB HTML | **2** | opus1, 2026-02-22 |
| `dynamo-deep-dive` | docs | 208KB HTML | **2** | opus1, 2026-02-22 |
| `nccl-user-guide` | docs | multi-page (index + subpages) | **2** | opus1, 2026-02-22. Tier 1 expanded to full TOC sub-pages; `py-modindex.html` returned 404 at fetch time. |
| `pytorch-cuda-semantics` | docs | 378KB HTML | **2** | opus1, 2026-02-22 |
| `cuda-graphs-guide` | docs | 312KB HTML | **2** | opus1, 2026-02-22 |
| `pytorch-pipelining-api` | docs | 582KB HTML | **2** | opus1, 2026-02-22 |
| `streamdiffusionv2` | repo+paper | 42KB HTML + repo (42 files) | **2** | codex2+opus2, 2026-02-22. Structured code dump (9597 lines, streamv2v/ first) |
| `gpipe` | paper | 539KB PDF | **2** | opus1, 2026-02-22. 11 pages, 5 tables, 49 refs |
| `pipedream-2bw` | paper | 2.2MB PDF | **2** | opus1, 2026-02-22. 14 pages (10+appendix), Algorithm 1, cost model |
| `zero-bubble-pp` | paper | 649KB PDF | **2** | opus1, 2026-02-22. 19 pages (12+appendix A-H), 12 tables, ILP formulation |
| `pipedit` | paper | 3.9MB PDF | **2** | gpt-xhigh, 2026-02-22. 10pp + supplementary, dense tables as monospaced blocks |
| `pagedattention` | paper | 1.5MB PDF | **2** | opus2, 2026-02-22. 16 pages, all equations/tables/64 refs |
| `causvid` | paper | 14.6MB PDF | **1** | opus3, 2026-02-22. New addition — AR blueprint for Wan 2.1. Tier 2 → gpt-xhigh. |

## Tier 2 policies (updated per Codex review)

These override the original CLAUDE.md definitions where they conflict. Update CLAUDE.md to match.

- **HTML docs/blogs**: Strip site chrome (nav, footer, sidebar) only. Keep **all** article body content verbatim. No selectivity — no "include relevant sections." Consistent policy across all docs resources.
- **Papers (PDF)**: Opus reads PDF pages directly (page by page). Add `conversion_notes:` field to YAML frontmatter documenting any equation/table conversion artifacts. This is the highest-quality approach but expensive.
- **GitHub repos**: Tier 2 = file tree + verbatim content under `## File: path/to/file.py` headings. **No interpretation, no architecture overview** — that's Tier 3. The `full.md` is a structured dump that Tier 3 cites. Update CLAUDE.md repo Tier 2 definition accordingly.
- **GitHub issues/RFCs**: Direct formatting from JSON structured data.

## Task flow (4 tracks, partially parallel)

### Track A: Implementation context expansion — DONE
**Status**: Complete. All 25 topic stubs have populated `## Implementation context` sections (topics 1-10,17 have brief paragraphs; topics 11-24 have 5-8 lines with specific working notes findings). Bridge doc (`refs/implementation-context.md`, 259 lines) is complete. Remaining work: `## Synthesis` sections are intentionally empty — that's Track D.

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

Order by bringup phase (synced with `refs/reading-guide.md`):

**Wave C1 — PP0 foundations (contracts work, no overlap, no TP):**
1. `nccl-user-guide` — group-call ordering, env vars, troubleshooting. PP0 hang prevention.
2. `pytorch-cuda-semantics` — streams/events mental model for P2P transport
3. `making-dl-go-brrrr` *(sharpen existing card)* — performance mental model grounds everything

**Wave C2 — PP0+overlap + PP0+recompute (streams, KV cache):**
4. `cuda-graphs-guide` — CUDAGraph capture for overlap phase
5. `causvid` — causal DiT framing + cache/recompute (PP0+recompute blocker)
6. `streamdiffusionv2` — rolling caches, Stream Batch, most load-bearing system reference
7. `pagedattention` — KV-cache-as-memory-management patterns

**Wave C3 — PP1 (TP inside mesh, compile survives):**
8. `funcol-rfc-93173` — compile-aware collectives for mesh_pg
9. `dynamo-deep-dive` — graph breaks, SymInt, guards
10. `ezyang-state-of-compile` — compile + distributed interaction

**Wave C4 — PP scheduling theory + API:**
11. `gpipe` — foundational PP concept
12. `pipedream-2bw` — 1F1B schedule
13. `zero-bubble-pp` — advanced PP scheduling (F/B/W split)
14. `pipedit` — pipelined sequence parallelism for DiT
15. `pytorch-pipelining-api` — PP schedule API

Also in Wave C1: sharpen `dit-paper` actionables using `implementation-context.md` guide table.

Note: `causvid` needs Tier 3 stub created first (Track H item).

### Track D: Topic synthesis docs (last, after C)
**Dependencies**: Tier 3 cards for the topic's resources must exist
**Effort**: ~1-2 sessions, can be incremental

Fill in the synthesis sections of `refs/topics/*.md` — mental model, key concepts, cross-resource agreement/disagreement, practical checklist, experiments to run. This is the highest-judgment work and only makes sense once the underlying resource cards are populated. PP-critical topics first: 13, 11, 16, 24, 22.

## Execution order

```
Tracks A+B: DONE (all 15 Phase 1 resources at Tier 2+)

Next:
  Wave C1 (any agent + user review): nccl-user-guide, pytorch-cuda-semantics
    + sharpen making-dl-go-brrrr + dit-paper
  Wave C2 (any agent + user review): cuda-graphs-guide, causvid, streamdiffusionv2, pagedattention
  Wave C3 (any agent + user review): funcol-rfc, dynamo-deep-dive, ezyang
  Wave C4 (any agent + user review): gpipe, pipedream-2bw, zero-bubble-pp, pipedit, pipelining-api
  Track D: topic synthesis for PP-critical topics (13, 11, 16, 24, 22)
  Track H: ongoing housekeeping (check at session start)
```

## Assignment guidance

| Track | Agent | Why |
|---|---|---|
| Track A (context expansion) | **DONE** | All 25 topic stubs + bridge doc populated. |
| Track B, B1 (HTML Tier 2) | **DONE** | Completed by opus (2026-02-22). |
| Track B, B3 (repo Tier 2) | **DONE** | Completed by gpt-xhigh+opus (2026-02-22). |
| Track B, B2 (Paper Tier 2) | **DONE** | Completed by opus (4 papers) + gpt-xhigh (pipedit). All 5 B2 papers at Tier 2. |
| Track C (Tier 3) | **Any agent**, user reviews each card | Any model can draft cards when asked. Quality gate is user review, not model selection. |
| Track D (topic synthesis) | User + **any agent** | Highest-judgment work; user's understanding matters most |

### Track C review workflow
Any agent can write a Tier 3 card when asked. The quality gate is **user review**, not model assignment. Workflow:
- Agent writes the card, referencing `full.md` + `refs/implementation-context.md` for grounding
- Agent presents card to user for review — do **not** mark `condensed` in manifest
- User can: approve → mark `condensed` | request changes → agent revises | add context

This keeps cards grounded without bottlenecking on a specific model.

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

## Key artifacts and their roles

| Artifact | Role | Consumers |
|---|---|---|
| `distributed_video_dit_inference.md` | **Canonical catalog** — exhaustive resource list (24 topics, ~85 resources). Read-only. | Manifest derivation, agents adding resources |
| `refs/manifest.yaml` | **Status tracking** — source of truth for resource pipeline state | All agents |
| `refs/reading-guide.md` | **Study guide** — spine (~15 resources) + bringup-synced reading order + topic index | Human learner, study sequencing |
| `PLAN.md` | **Agent coordination** — tracks, assignments, progress | All agents |

The reading guide is *not* a replacement for the catalog — it's a curated subset with phase-specific sequencing. The catalog remains the place to look when the reading guide says "see canonical catalog for full list."

## Manifest candidates (from reading guide reconciliation)

These resources appeared in the GPT-5.2 integrated learning path but are not in the manifest. Evaluate for addition.

**Strong candidates (Phase 1):**
- **Wan 2.1** (arxiv 2503.20314) — the target model. Referenced in PipeDiT bibliography; the canonical catalog never links to it.
- **CausVid** (arxiv 2412.07772) — causal DiT framing + KV cache/recompute. Directly relevant to PP0+recompute phase. URL found in StreamDiffusionV2 source.
- **StreamDiffusionV2 arxiv** (arxiv 2511.07399) — the actual paper. Manifest currently has project page + GitHub only; add arxiv URL.

**Phase 2 candidates (supplemental):**
- Merlyn Wang "NCCL Allreduce" — ring all-reduce intuition
- Zach DeVito caching allocator writeup — deeper than PyTorch docs
- NERSC roofline guide — pedagogical complement to NVIDIA's denser doc
- DDIA (schema evolution chapters) — relevant to PPEnvelope versioning
- Patterns of Distributed Systems / Idempotent Receiver (Fowler) — distributed patterns

## Cleanup

- [ ] Delete `distributed_video_dit_inference_integrated_learning_path.md` (superseded by `refs/reading-guide.md`)

## What NOT to do yet

- Phase 2 resources (medium/low priority) — only backfill when implementation pressure hits
- Topic synthesis for non-PP-critical topics — defer until the PP-critical topics are done
- `link_only` resources — skip unless they become load-bearing
