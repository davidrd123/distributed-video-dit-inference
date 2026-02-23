Here’s a nit-picky vet of `pp0-a4b-real-decode-codex-instructions.md` against what the repo actually does today (PP0 pilot A4a state + pipeline/server/VAE code).

## What’s solid and matches the code

1. **The early vs steady-state context-frame rule is correct.**
   Your conditional
   `((current_start_frame - num_frame_per_block) < kv_cache_num_frames)`
   matches `get_context_frames()` in `recompute_kv_cache.py` and the reference `_tp_v11_build_context_frames_override()` in `frame_processor.py`.

2. **The tensor layout warning for `encode_to_latent` is correct.**
   `encode_to_latent()` expects `[B, C, T, H, W]` and returns `[B, T, C, H, W]`. Your “permute before encode” call is exactly what `_tp_v11_build_context_frames_override()` does.

3. **Your decoded/context buffer max sizes match the production block.**

   * `context_frame_buffer_max_size = kv_cache_num_frames - 1`
   * `decoded_frame_buffer_max_size = 1 + (kv_cache_num_frames - 1) * 4`
     This matches `RecomputeKVCacheBlock`’s chunk-0 initialization and is consistent with the “first frame is special, then groups of 4” temporal structure implied by the VAE.

4. **Using `use_cache=True` for decode and `use_cache=False` for the one-shot anchor re-encode is aligned with the actual VAE wrapper.**
   `encode_to_latent(..., use_cache=False)` really does an explicit temporary-cache encode path. `decode_to_pixel(..., use_cache=True)` is the streaming path.

5. **Keeping rank0-side buffers separate from pipeline state is a reasonable design choice** for the pilot, since rank0 is not executing modular blocks, just doing post-processing and envelope construction.

## Things that are stale, misleading, or internally inconsistent

### 1) You now have two contradictory “where A4b lives” stories

* `pp0-a4-recompute-codex-instructions.md` (A4a doc) explicitly says A4b “happens when PP integrates into `frame_processor.py`, not in the pilot.”
* This A4b doc says “implement A4b in the pilot.”

That’s not just “difference in emphasis”; it changes what gets built. If the plan changed, you need to update A4a’s “pilot exits here” narrative (or rename this doc to something like **“A4b-pilot”** so readers don’t assume it’s the same A4b promised elsewhere).

### 2) The “semantic equivalence” paragraph is half-true and half-confusing

You say “A4b does NOT validate visual output quality vs single-rank baseline; that requires server integration.” That’s true *for the pilot* because it doesn’t render/compare frames.

But `pp-next-steps.md` Step A4 explicitly frames the goal as “Compare output quality to TP baseline.” So your A4b doc should say:

* “Pilot-A4b validates context-frames plumbing and numerical stability.”
* “Baseline visual equivalence is still required later (server integration).”

Right now, the doc reads like A4b is “done” without that equivalence gate. That’s misleading.

### 3) The cache-clearing advice references functions that don’t exist (in the code you cited)

You wrote:

> “See wan.py clear_cache_decode / clear_cache_encode.”

But the wrapper shown in `wan.py` exposes **`clear_cache()`**, not `clear_cache_decode` / `clear_cache_encode`. In this codebase, cache reset is `WanVAEWrapper.clear_cache()` which sets `model.first_batch = True`.

The fallback code you propose (`rank0_vae.model.clear_cache_decode`) is likely stale and can be removed. If you keep a fallback, at least name it as “maybe exists on older branches”, not “mirrors pipeline behavior”.

### 4) You’re missing the biggest practical gotcha: `decode_to_pixel()` forcibly `.to("cuda")`

`WanVAEWrapper.decode_to_pixel()` does:

```py
zs = zs.to(torch.bfloat16).to("cuda")
```

That’s not “move to `vae_device`”. It’s “move to the current default CUDA device”.

In distributed multi-GPU, this is fine **only if** your process has already set the correct CUDA device (usually via `torch.cuda.set_device(local_rank)` in your distributed init).

Your doc should explicitly warn:

* “Make sure distributed runtime sets the CUDA device before calling `decode_to_pixel`, or it may silently hop to cuda:0.”

This is a real footgun for PP bringup.

### 5) The “rank0 can’t build envelope k+1 until decode completes” statement is too broad

It’s only strictly true in **steady state + recompute chunks** (when you need the re-encoded anchor).

* In early regime `(current_start_frame - nfpb) < kv`, you can build `context_frames` from `first_context_frame + context_frame_buffer` without decoding.
* Also when `do_kv_recompute=False` (R1 or skipped chunks in R0b), you don’t need context frames at all, so you can prefetch while decoding.

Your doc *does* explain early vs steady-state, but later it collapses into “always can’t proceed until decode,” which is not accurate.

### 6) You don’t mention the PP0 overlap implementation detail that matters for correctness: waiting on the comm-stream event before decoding

In `pp0_pilot.py` overlap mode, the comms thread runs `pp.send_infer()`/`pp.recv_result()` on a separate CUDA stream and passes back a `ready_evt`. The main thread must:

* `wait_event(ready_evt)` **before touching** `res.latents_out` (and definitely before feeding it into VAE decode)

Your A4b doc currently doesn’t call this out. It’s easy to “follow the doc” and accidentally decode on a tensor that’s not stream-synchronized.

### 7) The “first_context_frame mismatch” subsection contains a contradictory sentence

You wrote:

> “The pipeline’s first_context_frame is set BEFORE denoise … But PrepareContextFramesBlock sets it from latents which is the DENOISED output.”

The first clause is wrong/misleading. In the code you cite, `PrepareContextFramesBlock` sets `first_context_frame` from `latents` (which is denoised at that point). There’s no “set before denoise” happening there (unless some other block does it, which your note doesn’t show and your argument doesn’t need).

This reads like leftover internal uncertainty. Delete the incorrect sentence and keep the correct one.

## Pass criteria: mostly fine, but you’re missing two checks that matter

Your “expected == observed generator calls” criterion is still the right primary invariant.

Two extra pass checks I’d add (because they catch real bugs quickly):

1. **Assert context frame shape when recompute is predicted**
   For any chunk where `_predict_do_kv_recompute(...)` is True, assert:

   * `context_frames.shape == (1, kv_cache_num_frames, 16, H_lat, W_lat)`
   * `context_frames.is_contiguous()` (or at least `.contiguous()` before send)

   This catches off-by-one buffer assembly and accidental pixel/latent confusion.

2. **Assert the override actually arrives and is used**
   Because `RecomputeKVCacheBlock` has the PP tripwire (“missing override under PP → crash”), you already get a hard fail if it’s missing.
   But you should still state explicitly in the doc: **“Always pass `context_frames_override` in `tp_worker_infer` kwargs, even when None, to prevent stale override reuse.”**
   That’s a pipeline behavior gotcha, not a recompute behavior gotcha.

## Concrete doc edits I’d make (minimal but fixes the problems)

1. **Rename / clarify scope at top**

   * “A4b (pilot): rank0 real decode + re-encode to produce context_frames_override”
   * “Server integration still required for visual baseline equivalence.”

2. **Replace the stale cache-clear reference**

   * Replace “clear_cache_decode/encode” with “WanVAEWrapper.clear_cache()”.

3. **Add a “Device gotcha” callout**

   * Mention `decode_to_pixel` does `.to("cuda")` and depends on correct default device.

4. **Add an “Overlap ordering” callout**

   * In overlap mode: for recompute steady-state chunks, decode+buffer update must happen before building the next recompute envelope.
   * Still allow prefetch for non-recompute chunks (R1 or skipped in R0b) if you want.

5. **Add an explicit “wait_event before decode” note**

   * “Main thread must wait on recv_item.ready_evt before decoding res.latents_out.”

6. **Delete the contradictory first_context_frame sentence**

   * Keep only: “PrepareContextFramesBlock sets first_context_frame from denoised latents when current_start_frame==0, so rank0 using result.latents_out[:, :1] is consistent.”

---

If you want, I can also propose a short “diff-style” patch for the markdown (just the text changes, no code).
