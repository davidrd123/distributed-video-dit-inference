Here’s a concrete G1 plan that matches the **current code you actually have** (and calls out where the repo doesn’t yet have the “stage0” object the deep-research prompt mentions).

Two grounding facts from current code:

* **PP1 server role split is already real**: rank0 runs FastAPI; rank1 is the PP1 mesh leader; ranks≥2 are TP workers (`scope-drd/src/scope/server/app.py` role dispatch, lines **4390–4527**).
* **Rank0 does not load the pipeline under PP1** (“PP1 rank0: skipping pipeline load”, `app.py` lifespan, lines **220–559**). So G1’s “stage0 stack” must be **explicit rank0-side VAE + text encoder**, not “reuse pipeline components”.

---

## 1) VAE loading plan (Q1)

### Can we instantiate `WanVAEWrapper` standalone?

Yes. The wrapper is self-contained: it loads weights, registers mean/std buffers, and exposes `encode_to_latent`, `decode_to_pixel`, `clear_cache` (`scope-drd/src/scope/core/pipelines/wan2_1/vae/wan.py`, lines **295–509**). There’s no dependency on `PipelineState`, scheduler, generator, etc.

### Exact load path you should follow

Mirror what `KreaRealtimeVideoPipeline` does when it loads the VAE, but without loading the generator or pipeline:

* Pipeline loads VAE like:
  `vae = WanVAEWrapper(model_name=base_model_name, model_dir=model_dir, vae_path=vae_path)` then `vae.to(device=device, dtype=dtype)`
  (see `scope-drd/src/scope/core/pipelines/krea_realtime_video/pipeline.py`, “Load vae” section).

So Stage0 should:

* Determine `base_model_name` from the same config as mesh ranks (`model.yaml` has `base_model_name: Wan2.1-T2V-14B`).
* Instantiate `WanVAEWrapper(...)`.
* Move to rank0 GPU (`cuda:0`) + `torch.bfloat16` (same as pipeline default).

### VRAM footprint estimate (A100, bf16)

You can put real numbers on the **weight** side:

* The published Wan2.1 VAE checkpoint `Wan2_1_VAE.pth` is **~254 MB**. ([Hugging Face][1])
  In bf16, GPU residency for weights is basically that order of magnitude (plus small overhead).

The rest is runtime working memory:

* Decoder/encoder internal feature caches (WanVAE does streaming; it keeps cache state unless `clear_cache()`).
* Intermediate activations/workspaces for 3D conv stacks.

A reasonable budgeting number for “VAE-only on rank0” is:

* **Weights**: ~0.25–0.35 GB
* **Working/caches**: ~0.5–1.5 GB
* **Total**: **~1–2 GB** on A100$_{70%}$

Even if this is off by 2×, it’s still nowhere near your generator footprint.

### Streaming vs batch mode (what to call)

Use the wrapper exactly the way it was designed:

* **Decode:** `decode_to_pixel(latents, use_cache=True)`

  * That calls `self.model.stream_decode(...)` and preserves cache state (`wan.py` lines **427–469**).
* **Re-encode (only if you implement “real context frames”):** `encode_to_latent(pixel, use_cache=False)`

  * `use_cache=False` uses a *temporary cache* (`_encode_with_cache`) and does **not** disturb streaming state (`wan.py` lines **342–420**).

**Cache lifecycle rule** (copying `pp0_pilot` intent):

* On `init_cache` / hard cut: call `vae.clear_cache()` (WanVAE sets `model.first_batch=True`, `wan.py` lines **471–475**).
  `pp0_pilot.py` uses `_maybe_clear_vae_cache()` at chunk0 before streaming decode (lines **565–863**).

### Should rank0 load text encoder at the same time?

For G1, yes if you choose prompt routing (a) (recommended below). If you don’t, you can stage it:

* **G1a:** VAE-only decode first (get pixels working).
* **G1b:** add text encoder once decode is stable.

---

## 2) Prompt routing recommendation (Q2)

Pick **(a) rank0 encodes text → sends `conditioning_embeds` in `PPEnvelopeV1`**.

### Why (a) is the right G1 choice

* **Contract already supports it.** `PPEnvelopeV1` already has `conditioning_embeds: torch.Tensor` and the mesh leader already routes it into `conditioning_embeds_override` with `tp_v11_conditioning_override=True` (`scope-drd/src/scope/server/pp1_mesh_leader.py`, `_pp1_unpack_to_call_params`).
* **You avoid putting UMT5 on the mesh critical path.** With (b), mesh has to run `TextConditioningBlock` encoding on prompt updates, blocking Phase B.
* **Prompt update latency target (“1–2 chunks”) is easiest with (a).** Rank0 can encode at chunk boundary and the next envelope carries the new tensor.

### VRAM reality check for UMT5-XXL encoder

HuggingFace lists “model size: **6B parameters**” for encoder-only UMT5-XXL. ([Hugging Face][2])
In bf16 that’s ≈ **12 GB** just for weights (6e9 × 2 bytes), plus activations/workspace. Budget **~13–16 GB** on GPU$_{75%}$.

On A100-80GB this is fine; on smaller cards it’s not.

### How to route prompt updates correctly (without re-implementing blending)

You do **not** need `EmbeddingBlendingBlock` on rank0 for G1 because:

* You’re not doing transitions.
* The mesh already consumes overrides cleanly:

  * `TextConditioningBlock` will skip local encoding when `tp_v11_conditioning_override=True` and `conditioning_embeds_override` is a tensor (`scope-drd/.../text_conditioning.py`, `tp_v11_override` path).
  * `EmbeddingBlendingBlock` will treat the override as an “immediate conditioning change” and set `conditioning_embeds_updated=True` internally, which triggers cross-attn cache reset downstream (`embedding_blending.py` docstring + override path).

But in PP1 you can’t directly set `conditioning_embeds_updated`; you must drive cache resets via the PP contract:

* When prompt changes, set **`reset_crossattn_cache=True`** in the envelope.
* Do **not** set `init_cache=True` unless you actually want to reset everything.
* This matches the intent in `EmbeddingBlendingBlock` docs: `conditioning_embeds_updated=True → resets cross-attn cache only`, while `init_cache=True → resets all caches`.

Mesh leader already maps:

* `reset_crossattn_cache` → `call_params["reset_crossattn_cache"]`
* `init_cache` is only `env.init_cache or env.reset_kv_cache` (so cross-attn reset doesn’t implicitly nuke KV).

### Pseudocode for (a)

This is the minimal “rank0 owns prompt encode” loop:

```py
# rank0 persistent state
last_prompts = None
cached_conditioning = None  # torch.Tensor on GPU (bf16)

def maybe_encode_prompts(prompts):
    nonlocal last_prompts, cached_conditioning
    if prompts == last_prompts and cached_conditioning is not None:
        return cached_conditioning, False  # no change

    # normalize: accept str / list[str] / list[dict{text,weight}]
    texts = normalize_to_text_list(prompts)  # G1: probably just str

    with torch.inference_mode():
        out = text_encoder(text_prompts=texts)["prompt_embeds"]  # [B, seq, dim]
        conditioning = out[:1].to(device=cuda0, dtype=torch.bfloat16).contiguous()

    last_prompts = prompts
    cached_conditioning = conditioning
    return conditioning, True  # changed

def build_envelope(chunk_index, prompts, ...):
    conditioning, changed = maybe_encode_prompts(prompts)

    env = PPEnvelopeV1(
        action=INFER,
        chunk_index=chunk_index,
        height=H, width=W,
        current_start_frame=chunk_index * latent_frames,
        do_kv_recompute=do_kv_recompute(...),
        num_denoise_steps=len(step_list),
        expected_generator_calls=(1 if do_kv_recompute else 0) + len(step_list),
        init_cache=(chunk_index == 0),
        reset_crossattn_cache=changed,     # key bit
        reset_kv_cache=False,
        conditioning_embeds=conditioning,  # key tensor
        latents_in=randn_latents(...),
        current_denoising_step_list=step_t,
        context_frames=(ctx if do_kv_recompute else None),
        pp_disable_vace=True,
    )
    env.validate_before_send()
    return env
```

### What you “lose” by skipping blending in G1

* Weighted prompt mixing (`list[dict{text,weight}]`) *could* be ignored at stage0 if you only pass one prompt.
  If you care: you can implement a minimal version of `TextConditioningBlock._normalize_prompts()` (it’s tiny) and encode multiple texts as a batch, then do blending on rank0 later.

### Does (a) constrain G2?

Not fatally.

* If you want transitions later, you have two options:

  1. Implement blending on rank0 (reuse `EmbeddingBlender` + `parse_transition_config`, and you’ll probably need to extend `PPEnvelopeV1` with “transition config” only if you want mesh to own it).
  2. Switch to a (b)-style contract where prompts/transition metadata are sent and mesh does TextConditioning+EmbeddingBlending. That’s a contract bump but not a rewrite.

G1 should not pre-optimize for G2. Get pixels + prompt updates working first.

---

## 3) Decode ∥ mesh inference overlap (Q3)

Pick **(b) Stage0 owns overlap internally**, i.e. hide the A3 machinery behind a small “stage0 API” so `FrameProcessor` doesn’t turn into a threading demo.

This directly mirrors the proven A3 pattern in `pp0_pilot.py`:

* comms thread
* bounded queues
* separate CUDA stream
* CUDA events + `record_stream()` for allocator safety
  (`pp0_pilot.py` lines **1016–1232**).

### Threading model

**Comms thread (rank0):**

* Owns **all** PPControlPlane send/recv calls (`pp.send_infer`, `pp.recv_result`).
  This is important: don’t sprinkle `torch.distributed` across threads.

* Runs on `comm_stream = torch.cuda.Stream()` and:

  * waits for `ready_evt` from main stream (so tensors are ready),
  * sends envelope,
  * receives result,
  * records `done_evt` on comm_stream,
  * puts `(result, done_evt)` into `recv_q`.

**Main thread (FrameProcessor loop):**

* Builds envelope for chunk N+1
* Enqueues it to `send_q`
* Dequeues result for chunk N (if available)

  * `main_stream.wait_event(done_evt)`
  * `res.latents_out.record_stream(main_stream)` (exactly like A3)
* Decodes latents_out → pixels via VAE
* Emits frames to WebRTC

### The big gotcha: recompute context frames can kill overlap

Under PP, `RecomputeKVCacheBlock` refuses to build context frames on mesh ranks and will throw if `context_frames_override` is missing (`scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` has the explicit PP guard and RuntimeError).

So stage0 must supply `env.context_frames` whenever `do_kv_recompute=True`. Also, if you build “real context frames” (A4b style), you introduce a dependency on having decoded/updated buffers before you can build the next envelope, which reduces overlap.

For G1, I’d do this:

* **Use the pp1_pilot synthetic context frame builder** for recompute:

  * `_build_synthetic_context_frames(prev_latents_out, kv_cache_num_frames)` (`scripts/pp1_pilot.py`, lines **286–447**).
  * This lets you enqueue envelope N+1 immediately after receiving latents_out(N), *before* decoding it, so decode(N) overlaps with mesh compute(N+1).

Then you can later add the A4b “real context frames” path as an option/flag (see pp0_pilot `_build_real_context_frames`, lines **565–863**) once streaming is stable.

### Do you really need CUDA Events and `record_stream()` in the server?

Yes if you do multi-stream (and you should, to avoid comm stalling decode).

The exact allocator hazard is documented in A3:

* Comms allocates/receives tensors on `comm_stream`
* Main stream uses them for decode
* Without `record_stream(main_stream)`, CUDA ordering is not enough to stop the caching allocator from recycling that memory early
  (`pp0_pilot.py` lines **1150–1232** explain this explicitly and do the right thing).

### Decode latency estimate

I don’t have a measurement from your repo logs. Based on:

* VAE weights are only ~254MB([Hugging Face][1])
* resolution 320×576 and chunk size `num_frame_per_block=3` (`model.yaml`)
* WanVAE’s temporal factor means decode likely emits more than 3 pixel frames per chunk (the buffer sizing in pp0_pilot implies `1 + (kv-1)*4` pixel frames for kv=3, i.e. 9 frames)

I’d bet:

* **decode(1 chunk)** on A100 is **~20–80 ms**$_{60%}$.

But you should measure with CUDA events around `vae.decode_to_pixel()`; don’t guess in production.

---

## 4) `prepare()` plan (Q4)

### G1 (text-only, no video input): keep skipping `prepare()`

Yes: the current server already treats `Requirements` primarily as **input buffering gating** (see `frame_processor.py` around lines **4080–4225** where video input/control maps are assembled only when needed). If PP1 G1 doesn’t ingest video frames, skipping `prepare()` is fine.

What you **must** replicate on rank0 stage0 (because pipeline isn’t there):

* `height`, `width` validation: Krea pipeline requires `scale_factor=16` (VAE 8 × patch embed 2) (see `KreaRealtimeVideoPipeline.__init__` comments and `validate_resolution(...)` call).
* Default denoising schedule: `DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]` in `krea_realtime_video/pipeline.py`.
* Default `kv_cache_attention_bias`: `DEFAULT_KV_CACHE_ATTENTION_BIAS` env fallback is 0.3 (`pipeline.py` top).
* `noise_scale`, `base_seed` handling.

Those become stage0’s job because rank0 has no pipeline state under PP1 (`app.py` lines **220–559**).

### G2 (video/VACE): you’ll need a real “prepare equivalent”

For V2V you’ll need:

* an input-size contract (how many frames to buffer per chunk),
* mode resolution and parameter normalization (what `prepare_for_mode` + `resolve_input_mode` do today inside the pipeline).

I would not build this in G1. The lowest-risk plan for G2 is:

* Add a small `Stage0Requirements` object that at least returns `input_size` and any VACE input sizing info.
* Eventually, either:

  * reuse `prepare_for_mode(...)` logic without loading the generator, or
  * actually load a lightweight “config-only” pipeline object on rank0 (no generator weights) purely to reuse prepare semantics.

---

## 5) Implementation ordering

Here’s a sane order that doesn’t entangle everything at once:

1. **G1.0 Pixels first (VAE-only)**

   * Load `WanVAEWrapper` on rank0.
   * Decode `PPResultV1.latents_out` and stream pixels to WebRTC.
   * Keep prompts fixed (even if mesh still uses env var).
   * This is a tight loop; easiest to debug.

2. **G1.1 Dynamic prompts (rank0 text encoder → `conditioning_embeds` in envelope)**

   * Load `WanTextEncoderWrapper` on rank0 and cache `conditioning_embeds`.
   * Set `reset_crossattn_cache=True` only when prompt changes.
   * Mesh leader already expects override tensors (`pp1_mesh_leader.py`), so this is mostly rank0 work.

3. **G1.2 Overlap (A3-style comms thread)**

   * Add comms thread + bounded queues + comm stream + events + `record_stream()`.
   * Keep recompute context frames **synthetic** for now (pp1_pilot helper), so you can always enqueue next envelope before decode.

4. **G1.3 Quality upgrade (optional): real context frames (A4b)**

   * If/when drift matters: implement `_build_real_context_frames` using VAE re-encode like pp0_pilot lines **565–863**.
   * Accept that recompute chunks may reduce overlap unless you tune recompute frequency.

5. **G2 prep plumbing** (defer)

   * Only then worry about `prepare()` equivalence and video input buffering.

---

## 6) Red flags (things that will bite you)

1. **PP recompute hard-fails if you don’t supply context frames**

   * `RecomputeKVCacheBlock` explicitly raises under PP if `context_frames_override` is missing (`recompute_kv_cache.py` checks `SCOPE_PP_ENABLED==1` and throws).
   * Therefore: if `env.do_kv_recompute=True`, `env.context_frames` must be a tensor, every time.
   * Also: `PPEnvelopeV1.validate_before_send()` will fail if `do_kv_recompute=True` but `context_frames is None` (`pp_contract.py`).

2. **Mesh leader currently sets `prompts=""` and forces override mode**

   * In `_pp1_unpack_to_call_params`, `prompts` is always `""` and `tp_v11_conditioning_override=True`.
   * If stage0 ever sends `conditioning_embeds=None`, the mesh will either:

     * encode an empty prompt, or
     * do something unexpected depending on block ordering.
   * So: stage0 must always send a real conditioning tensor (or you add a fallback path in mesh leader).

3. **Cache reset semantics: don’t accidentally nuke KV**

   * For prompt updates you want **cross-attn reset only**:

     * set `reset_crossattn_cache=True`
     * don’t set `init_cache=True`
   * The difference matters (EmbeddingBlendingBlock docs spell it out).

4. **If you do overlap with multi-stream, you need `record_stream()`**

   * A3’s `record_stream()` is not optional decoration. It avoids allocator bugs that look like random corruption.

5. **`WanVAEWrapper.decode_to_pixel` unconditionally `.to("cuda")`**

   * In `decode_to_pixel`, it does `zs = zs.to(torch.bfloat16).to("cuda")` (wan.py lines **439–444**).
   * In PP1 rank0 that’s fine (you’re on CUDA), but it’s a footgun if you ever try CPU decode or non-default device routing.

6. **UMT5-XXL VRAM is real**

   * 6B params([Hugging Face][2]) ⇒ ~12GB bf16 weights, plus runtime overhead.
   * If you ever target smaller GPUs, you’ll need a CPU text-encoder fallback (but then prompt updates won’t hit “1–2 chunks” latency unless you hide the encode cost elsewhere).

---

If you want one crisp “G1 default policy” that’s internally consistent:

* **Stage0 loads VAE + text encoder on rank0 GPU**
* **Prompt updates:** encode on rank0, cache, send `conditioning_embeds`, set `reset_crossattn_cache=True`
* **Recompute:** keep enabled, but use **synthetic context frames from `prev_latents_out`** for G1 so overlap works
* **Overlap:** A3 comms thread + CUDA stream/events, one-chunk-ahead

That gets you: real pixels + dynamic prompts + no mesh-side prompt latency + good utilization.

(Also: unclench your shoulders and blink at something far away for 10 seconds.)

[1]: https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1_VAE_bf16.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1_VAE_bf16.safetensors"
[2]: https://huggingface.co/Able2/umt5-xxl-encode-only "https://huggingface.co/Able2/umt5-xxl-encode-only"
