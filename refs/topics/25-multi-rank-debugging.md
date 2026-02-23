---
status: draft
---

# Topic 25: Multi-rank debugging patterns

Debugging distributed PyTorch code is qualitatively different from single-process debugging. Failures are **non-local** (one rank's crash manifests as a timeout on another), **interleaved** (torchrun mixes stderr from all ranks), and **silent** (device mismatches or warmup side effects may not error until much later). This topic collects the practical patterns discovered during PP0 bringup — tribal knowledge that no existing topic or resource covers.

## Resources

<!-- No formal resources exist yet — this topic is sourced entirely from bringup experience -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-distributed-api | PyTorch Distributed API Reference | medium | converted |
| torchrun-elastic | torchrun / torch.distributed.elastic Reference | medium | pending |
| nccl-user-guide | NCCL User Guide | high | condensed |

## Implementation context

All five patterns below were discovered during Scope PP0 bringup (Steps A0–A2) on 2×H100 80GB with NVLink, running torchrun with `--nproc_per_node=2`. The patterns are documented in `refs/implementation-feedback.md` § 2 (patterns 2a–2e). The operator test matrix (`refs/operator-test-matrix.md`) tests for specific *planned* failure modes, but these patterns address the *unplanned* failures that consumed most debugging time.

See: `refs/implementation-feedback.md` § 2, `refs/operator-test-matrix.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`.

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/pp_control.py` (preflight + logging patterns)
- `scope-drd/src/scope/core/distributed/control.py` (TP control plane with shutdown/watchdog)

## Raw patterns from bringup

These are the five patterns from the feedback, presented as-is:

### Pattern A: Reading torchrun crash output (2a)
When rank1 crashes, torchrun kills rank0 too, and the output interleaves both ranks' stderr. The actual root cause is often buried 50+ lines above the `torch.distributed.elastic` summary. **Technique**: search backward from "RuntimeError" or "NCCL error" for the rank that crashed *first*. The other rank's error is usually "connection reset" or "NCCL timeout" — a symptom, not the cause.

### Pattern B: Device mismatches in multi-GPU code (2b)
`torch.cuda.current_device()` returns 0 on all ranks unless `torch.cuda.set_device(local_rank)` was called. This is not an NCCL problem — it's a CUDA context problem. Hit in `sinusoidal_embedding_1d()` which used `device=torch.cuda.current_device()` as a default arg. **Rule**: never use `torch.cuda.current_device()` in model code; use `tensor.device` from an input. Corollary: `torch.zeros(..., device="cuda")` also goes to device 0.

### Pattern C: Silent warmup failures (2c)
Pipeline constructors that run warmup inference silently break in multi-rank setups because warmup runs before the distributed coordination loop starts. Hit twice: first with TP (already gated), then with PP (had to add a gate). **Rule**: any "do work in `__init__`" is suspect in distributed code. Constructor should only allocate; first real inference should be driven by the coordination loop.

### Pattern D: Envelope/message debugging (2d)
When a p2p send/recv fails, you need to know what was *in* the envelope. Without preflight logging, a recv-side shape mismatch gives "expected [1, 2160, 20, 128] got [2160, 40, 128]" with no context about what the sender thought it was sending. **Rule**: log message contents before the commitment point, not after.

### Pattern E: Distinguishing hangs from crashes (2e)
A hang (all ranks blocked in NCCL) looks different from a crash (one rank exits, others timeout). `NCCL_DEBUG=INFO` helps but produces enormous output. **Technique**: set `NCCL_TIMEOUT` (or `SCOPE_DIST_TIMEOUT_S`) low during bringup (30–60s) so hangs surface quickly. Production can use longer timeouts.

## Synthesis

### Mental model

- **Distributed debugging is an exercise in “find the first divergence”**: the symptom you see (rank0 timeout, NCCL hang, connection reset) is usually downstream of the true root cause (rank1 threw, rank1 never entered a collective, rank1 is on the wrong CUDA device, rank0 sent a malformed envelope).
- Most of the “time sink” bugs in bringup were not deep algorithmic issues—they were *coordination* issues:
  - incorrect device placement on one rank,
  - work happening outside the coordination loop (constructor warmup),
  - missing preflight logging at commitment points,
  - long timeouts that hide the root cause for minutes.
- The operator posture is: **crash > hang**, and **shorten the feedback loop**:
  - fail before commitment points (Topic 20),
  - enforce env parity at init (Topic 02/12),
  - keep timeouts low during bringup (Topic 03),
  - write down message contents before send (Topic 20),
  - treat “first rank to error” as the root cause and ignore the cascade.

Source: `refs/implementation-feedback.md` §2 (patterns A–E), grounded in PP0 bringup A0–A2.

### Key concepts

- **Primary vs secondary failure**: the first rank to throw is the root cause; the other rank’s timeout is the symptom. In torchrun output, you must find the first exception, not the final elastic summary. (`refs/implementation-feedback.md` §2a.)
- **CUDA device context**: `torch.cuda.current_device()` is not rank-local unless you set it. In model code, prefer `tensor.device` and never default to `device="cuda"` in multi-rank code. (`refs/implementation-feedback.md` §2b.)
- **Coordination loop boundary**: constructors should allocate, not execute inference/warmup; warmup must be driven through the distributed coordination loop so all ranks participate identically. (`refs/implementation-feedback.md` §2c.)
- **Commitment point logging**: log envelope IDs + key schema fields before sending any header that would strand a peer (Topic 20; `refs/implementation-feedback.md` §2d.)
- **Timeouts as a debugging tool**: long NCCL timeouts turn “bug” into “10-minute stall.” During bringup, use 30–60s timeouts so hangs surface fast, then revisit production values. (`refs/implementation-feedback.md` §2e; Topic 03.)

### Cross-resource agreement / disagreement

- This topic is “mostly Scope-local,” but it’s consistent with the broader operator manuals:
  - Topic 20’s “validate/pickle/spec before header” is exactly Pattern D (“log message contents before commitment”) turned into an enforceable rule.
  - Topic 03’s crash-only + watchdog framing is exactly Pattern E (“distinguish hang vs crash” by shortening timeouts and exiting decisively).
  - Topic 02’s “same collectives, same order” explains why so many “it crashed on rank1” incidents present as “rank0 hung.”

### Practical checklist

When a distributed run fails, do this in order:

1. **Identify the first rank to fail**
   - Search for the earliest `RuntimeError` / Python traceback.
   - Treat “connection reset”, “NCCL timeout”, “recv failed” on the other rank as secondary symptoms.

2. **Confirm device discipline**
   - Ensure each rank called `torch.cuda.set_device(local_rank)` early.
   - Grep for `torch.cuda.current_device()` and any `device="cuda"` defaults in model code; replace with `tensor.device`.

3. **Confirm nothing runs “outside the loop”**
   - Any inference/warmup in a pipeline constructor is a red flag; gate it behind role checks or move it into the control plane.

4. **Make the commitment points loud**
   - Before every header send/broadcast, log `action`, `call_id`, `chunk_index`, `cache_epoch`, and the tensor manifest summary (`n_specs/n_tensors`, shapes/dtypes).

5. **Shorten the hang detection window**
   - Set bringup timeouts low (`SCOPE_DIST_TIMEOUT_S=30..60`) so “hang class” bugs don’t waste 10 minutes per iteration.

### Gotchas and failure modes

- **“Rank0 hung” is rarely the root cause**: it’s often rank1 threw (or diverged) and rank0 is just waiting in a collective/recv.
- **Logging can change behavior**: printing inside compiled regions can introduce graph breaks; keep hot-path debug logging outside compiled regions (Topic 09/12).
- **Per-rank device bugs masquerade as distributed bugs**: a tensor accidentally created on `cuda:0` on rank1 can show up as an NCCL/shape error later; treat device placement as a first-class invariant in bringup.

### Experiments to run

- **Timeout sanity**: run with a 30–60s distributed timeout and intentionally kill rank1; verify rank0 fails quickly and exits (not “wait forever”).
- **Device mismatch repro**: intentionally create a tensor with `device="cuda"` on rank1 and confirm your device discipline checks (or preflight) catches it with a clear message.
- **Constructor warmup gate**: add a test that ensures pipeline constructors do not execute inference when `WORLD_SIZE>1`; enforce via an explicit env or role gating.
