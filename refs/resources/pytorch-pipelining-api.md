# PyTorch torch.distributed.pipelining API

| Field | Value |
|-------|-------|
| Source | https://docs.pytorch.org/docs/stable/distributed.pipelining.html |
| Type | docs |
| Topics | 14 |
| Status | condensed |

## Why it matters

PyTorch’s built-in pipeline-parallel toolkit: a model-splitting frontend plus a distributed runtime with multiple PP schedules (GPipe, 1F1B, interleaved, looped, zero-bubble variants). If our PP0 pilot proves the Stage 0 / Stage 1 split and contracts (Steps A1–A5), this is the natural “productionization” path for standard schedules and microbatch plumbing, vs maintaining bespoke send/recv scheduling forever.

## Key sections

- [Step 1: build `PipelineStage`](../../sources/pytorch-pipelining-api/full.md#step-1-build-pipelinestage) — static-shape requirement, buffer allocation, and send/recv responsibilities.
- [Step 2: use `PipelineSchedule` for execution](../../sources/pytorch-pipelining-api/full.md#step-2-use-pipelineschedule-for-execution) — the rank-wise calling convention for `.step(...)`.
- [Option 1: splitting a model manually](../../sources/pytorch-pipelining-api/full.md#option-1-splitting-a-model-manually) — how to structure a model for easy partitioning.
- [Option 2: splitting a model automatically](../../sources/pytorch-pipelining-api/full.md#option-2-splitting-a-model-automatically) — `pipeline(...)`, `SplitPoint`, and what “automatic” means.
- [Technical Deep Dive](../../sources/pytorch-pipelining-api/full.md#technical-deep-dive) — how the `pipeline` frontend uses `torch.export` and reconstructs per-stage modules.
- [Implementing Your Own Schedule](../../sources/pytorch-pipelining-api/full.md#implementing-your-own-schedule) — single-stage vs multi-stage-per-rank schedule base classes.
- [Logging](../../sources/pytorch-pipelining-api/full.md#logging) — `TORCH_LOGS=+pp` / `pp` / `-pp`.
- [API Reference → Pipeline Schedules](../../sources/pytorch-pipelining-api/full.md#pipeline-schedules) — the concrete schedule classes and how they map to the papers.

## Core claims

1. **Claim**: `torch.distributed.pipelining` is a (currently alpha) toolkit that combines a splitting frontend with a distributed runtime to automate model partitioning, microbatch scheduling, communication, and (for training) gradient propagation.
   **Evidence**: [sources/pytorch-pipelining-api/full.md#pipeline-parallelism](../../sources/pytorch-pipelining-api/full.md#pipeline-parallelism), [sources/pytorch-pipelining-api/full.md#what-is-torchdistributedpipelining](../../sources/pytorch-pipelining-api/full.md#what-is-torchdistributedpipelining)

2. **Claim**: A `PipelineStage` allocates communication buffers, creates send/recv ops to peers, and manages intermediate buffers; it requires static input/output shapes and raises `PipeliningShapeError` if runtime shapes do not match the expected shapes.
   **Evidence**: [sources/pytorch-pipelining-api/full.md#step-1-build-pipelinestage](../../sources/pytorch-pipelining-api/full.md#step-1-build-pipelinestage)

3. **Claim**: Executing pipelined code is schedule-driven: you attach a `PipelineStage` to a `PipelineSchedule` (e.g., `ScheduleGPipe`), and call `schedule.step(...)` per rank; the first stage rank passes whole-batch inputs and other ranks call `step()` to participate.
   **Evidence**: [sources/pytorch-pipelining-api/full.md#step-2-use-pipelineschedule-for-execution](../../sources/pytorch-pipelining-api/full.md#step-2-use-pipelineschedule-for-execution)

4. **Claim**: The package supports two splitting frontends: (a) manual stage modules wrapped by `PipelineStage`, and (b) an automatic `pipeline(...)` frontend that uses `torch.export` to capture a full graph and then splits/reconstructs per-stage submodules while preserving forward behavior and activation dataflow.
   **Evidence**: [sources/pytorch-pipelining-api/full.md#options-for-splitting-a-model](../../sources/pytorch-pipelining-api/full.md#options-for-splitting-a-model), [sources/pytorch-pipelining-api/full.md#technical-deep-dive](../../sources/pytorch-pipelining-api/full.md#technical-deep-dive)

5. **Claim**: `SplitPoint` and `split_spec` allow inserting split points before/after execution of named submodules; alternatively `pipe_split()` can be used as an explicit boundary marker inside `forward()`.
   **Evidence**: [sources/pytorch-pipelining-api/full.md#model-split-apis](../../sources/pytorch-pipelining-api/full.md#model-split-apis)

6. **Claim**: The doc explicitly positions PP as applicable to large-model inference, and schedule/config signatures consistently allow `loss_fn=None`, separating pipeline execution mechanics from training-specific loss/backward semantics.
   **Evidence**: [sources/pytorch-pipelining-api/full.md#why-pipeline-parallel](../../sources/pytorch-pipelining-api/full.md#why-pipeline-parallel), [sources/pytorch-pipelining-api/full.md#pipeline-schedules](../../sources/pytorch-pipelining-api/full.md#pipeline-schedules)

7. **Claim**: Schedules are implemented as subclasses of `PipelineScheduleSingle` (one stage per rank) or `PipelineScheduleMulti` (multiple stages per rank), and the library ships concrete schedules including GPipe, 1F1B, Interleaved 1F1B, Looped BFS, and zero-bubble variants (Interleaved Zero Bubble, ZBV Zero Bubble).
   **Evidence**: [sources/pytorch-pipelining-api/full.md#implementing-your-own-schedule](../../sources/pytorch-pipelining-api/full.md#implementing-your-own-schedule), [sources/pytorch-pipelining-api/full.md#pipeline-schedules](../../sources/pytorch-pipelining-api/full.md#pipeline-schedules)

8. **Claim**: The package includes microbatch utilities for chunking positional/keyword args into microbatches and merging outputs back, via `TensorChunkSpec`, `split_args_kwargs_into_chunks`, and `merge_chunks`.
   **Evidence**: [sources/pytorch-pipelining-api/full.md#microbatch-utilities](../../sources/pytorch-pipelining-api/full.md#microbatch-utilities)

9. **Claim**: PP-specific logging is controlled by `TORCH_LOGS` categories (`+pp`, `pp`, `-pp`) via `torch._logging`.
   **Evidence**: [sources/pytorch-pipelining-api/full.md#logging](../../sources/pytorch-pipelining-api/full.md#logging)

## API surface / configuration

**Splitting frontend (build a `Pipe`):**
- `pipeline(module, mb_args, mb_kwargs=None, split_spec=None, split_policy=None)` — split a module given a `split_spec` and example microbatch inputs.
- `SplitPoint.{BEGINNING,END}` — split marker positions for `split_spec`.
- `pipe_split()` — explicit boundary marker inside `forward()` (no-op if run eagerly).
- `Pipe.get_stage_module(stage_idx)` — retrieve a per-stage `nn.Module` to wrap / checkpoint / compose with other parallelism.

**Stage runtime:**
- `PipelineStage(submodule, stage_index, num_stages, device, input_args=None, output_args=None, group=None, ...)`
- `build_stage(stage_module, stage_index, pipe_info, device, group=None)`

**Schedules (execution):**
- Single-stage-per-rank: `ScheduleGPipe`, `Schedule1F1B` (subclasses of `PipelineScheduleSingle`)
- Multi-stage-per-rank: `ScheduleInterleaved1F1B`, `ScheduleLoopedBFS`, `ScheduleInterleavedZeroBubble`, `ScheduleZBVZeroBubble` (subclasses of `PipelineScheduleMulti`)
- Common entrypoint: `PipelineSchedule{Single,Multi}.step(*args, target=None, losses=None, return_outputs=True, **kwargs)`

**Microbatch plumbing:**
- `TensorChunkSpec(split_dim)`
- `split_args_kwargs_into_chunks(args, kwargs, chunks, args_chunk_spec=None, kwargs_chunk_spec=None)`
- `merge_chunks(chunks, chunk_spec)`

**Debugging:**
- `TORCH_LOGS=+pp` / `pp` / `-pp`

Evidence: [sources/pytorch-pipelining-api/full.md#model-split-apis](../../sources/pytorch-pipelining-api/full.md#model-split-apis), [sources/pytorch-pipelining-api/full.md#pipeline-stages](../../sources/pytorch-pipelining-api/full.md#pipeline-stages), [sources/pytorch-pipelining-api/full.md#pipeline-schedules](../../sources/pytorch-pipelining-api/full.md#pipeline-schedules), [sources/pytorch-pipelining-api/full.md#microbatch-utilities](../../sources/pytorch-pipelining-api/full.md#microbatch-utilities), [sources/pytorch-pipelining-api/full.md#logging](../../sources/pytorch-pipelining-api/full.md#logging)

## Actionables / gotchas

- **Adoption path for us is “manual first, API later”**: follow the PP0 plan (Steps A1–A5) to validate contracts, queueing, overlap, and rank0-out-of-mesh topology with explicit send/recv before considering switching the execution loop to `torch.distributed.pipelining`. See: `refs/implementation-context.md` (Phase 3) and `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (Steps A1–A5).
- **Static shapes are a hard constraint**: `PipelineStage` requires static input/output shapes and throws `PipeliningShapeError` on mismatches. That aligns with our “same shapes, same code path” distributed contract, but it means any variable-length/video-conditional tensors must be normalized (padding/shape contracts) before using this API. See: [sources/pytorch-pipelining-api/full.md#step-1-build-pipelinestage](../../sources/pytorch-pipelining-api/full.md#step-1-build-pipelinestage).
- **Automatic splitting requires `torch.export` full-graph capture**: the `pipeline(...)` frontend uses `torch.export`; if the model is not “full-graph’able” (dynamic control flow / unsupported ops), you’ll need manual splitting (`PipelineStage` on hand-built stage modules). See: [sources/pytorch-pipelining-api/full.md#technical-deep-dive](../../sources/pytorch-pipelining-api/full.md#technical-deep-dive), [sources/pytorch-pipelining-api/full.md#hugging-face-examples](../../sources/pytorch-pipelining-api/full.md#hugging-face-examples).
- **Check the “sequential partition / no skip connections” assumption at stage boundaries**: the runtime assumes sequential partitioning (stage outputs feed stage inputs with no skip connections across stages). For our DiT PP split, keep boundaries on the main activation stream (contiguous block ranges) and avoid cross-stage side paths. See: [sources/pytorch-pipelining-api/full.md#pipeline-stages](../../sources/pytorch-pipelining-api/full.md#pipeline-stages).
- **Most schedules are documented in training language**: schedule names and docs describe forward+backward steady state. For inference adoption, validate that `loss_fn=None` gives a forward-only schedule that matches our streaming needs (e.g., microbatch-as-in-flight-chunk semantics) before committing. Evidence that inference is a target use case + `loss_fn=None` exists is in the doc, but the exact inference semantics should be proven with a small harness. See: [sources/pytorch-pipelining-api/full.md#why-pipeline-parallel](../../sources/pytorch-pipelining-api/full.md#why-pipeline-parallel), [sources/pytorch-pipelining-api/full.md#pipeline-schedules](../../sources/pytorch-pipelining-api/full.md#pipeline-schedules).
- **Use PP logging early**: `TORCH_LOGS=+pp` helps confirm microbatch chunking, schedule ordering, and stage send/recv behavior before you start tuning overlap.

## Related resources

- [gpipe](gpipe.md) -- ScheduleGPipe implements this paper's approach
- [pipedream-2bw](pipedream-2bw.md) -- Schedule1F1B implements the 1F1B schedule
- [zero-bubble-pp](zero-bubble-pp.md) -- ScheduleInterleavedZeroBubble and ScheduleZBVZeroBubble implement zero-bubble
- [pipedit](pipedit.md) -- pipeline-parallel DiT inference that may use these schedules
- [nccl-user-guide](nccl-user-guide.md) -- stream semantics + ordering constraints for any distributed send/recv/collective activity
