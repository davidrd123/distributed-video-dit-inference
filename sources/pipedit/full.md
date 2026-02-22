---
title: "PipeDiT: Accelerating Diffusion Transformers in Video Generation with Task Pipelining and Model Decoupling"
source_url: https://arxiv.org/abs/2511.12056
fetch_date: 2026-02-22
source_type: paper
authors:
  - Sijie Wang
  - Qiang Wang
  - Shaohuai Shi
arxiv_id: "2511.12056v1"
date: 2025-11-15
conversion_notes: |
  Converted by GPT-5.2 using rendered PDF page images (12 pages) and cross-checked
  against `pdftotext -raw` output for completeness. Equations are rendered in LaTeX
  notation. Wide tables are preserved as monospaced text blocks to avoid loss from
  reflow. Figures are described in brackets in addition to the original captions.
---

# PipeDiT: Accelerating Diffusion Transformers in Video Generation with Task Pipelining and Model Decoupling

Sijie Wang, Qiang Wang, Shaohuai Shi*

School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen

25b951105@stu.hit.edu.cn, {qiang.wang, shaohuais}@hit.edu.cn

arXiv:2511.12056v1 [cs.CV] 15 Nov 2025

\* Corresponding author.

## Abstract

Video generation has been advancing rapidly, and diffusion transformer (DiT) based models have demonstrated remarkable capabilities. However, their practical deployment is often hindered by slow inference speeds and high memory consumption. In this paper, we propose a novel pipelining framework named PipeDiT to accelerate video generation, which is equipped with three main innovations. First, we design a pipelining algorithm (PipeSP) for sequence parallelism (SP) to enable the computation of latent generation and communication among multiple GPUs to be pipelined, thus reducing inference latency. Second, we propose DeDiVAE to decouple the diffusion module and the variational autoencoder (VAE) module into two GPU groups, whose executions can also be pipelined to reduce memory consumption and inference latency. Third, to better utilize the GPU resources in the VAE group, we propose an attention co-processing (Aco) method to further reduce the overall video generation latency. We integrate our PipeDiT into both OpenSoraPlan and HunyuanVideo, two state-of-the-art open-source video generation frameworks, and conduct extensive experiments on two 8-GPU systems. Experimental results show that, under many common resolution and timestep configurations, our PipeDiT achieves 1.06× to 4.02× speedups over OpenSoraPlan and HunyuanVideo.

## 1 Introduction

Video generation models (Li et al. 2024a; Cho et al. 2024; Sun et al. 2024a; Blattmann et al. 2023) have advanced rapidly over the past two years. Such models take textual or visual inputs and synthesize a continuous video as output. Diffusion Transformers (DiT) (Fan et al. 2025; Ma et al. 2024; Yang et al. 2024) have emerged as the primary framework for video generation due to the high quality of the videos they produce. The key idea of DiT is to adopt a progressive denoising process (Ho, Jain, and Abbeel 2020; Sohl-Dickstein et al. 2015; Song and Ermon 2019): starting from pure noise, the model iteratively refines the signal through a reverse diffusion process until a high-quality video is generated as shown in Fig. 1. However, the inherently sequential nature of its reverse diffusion process severely restricts parallelism during inference (Li et al. 2023b; Shih et al. 2023; Chen et al. 2024a).

Current inference optimizations for DiT target both image and video generation (Liu et al. 2024; Zhang, Luo, and Lu 2024; Luo et al. 2025). For image generation, DistriFusion (Li et al. 2024b) accelerates inference by splitting the input into multiple patches and distributing them across different GPUs. It reuses intermediate feature maps from the previous timestep to provide context for the current step, and hides communication overhead via asynchronous communication in the computation pipeline. PipeFusion (Fang et al. 2024b) also divides images into patches and distributes network layers across multiple GPUs to address memory limitations during generation. For video generation, methods (Chen et al. 2024c; Selvaraju et al. 2024) like Tea-cache (Liu et al. 2025) analyze the correlation between features across adjacent timesteps and reuse outputs from the previous step to reduce the number of timesteps, thus improving inference efficiency (Ma, Fang, and Wang 2024; Zhao et al. 2024b). However, these approaches may theoretically introduce degradation in generation quality. Consequently, the majority of current video generation models utilize system-level optimizations such as sequence parallelism (SP) (Li et al. 2023a; Sun et al. 2024b; Zhao et al. 2024a) to expedite the generation process while preserving the quality of the generated videos.

Currently, two main SP paradigms have been proposed. The first is DeepSpeed-Ulysses (termed as Ulysses afterward) (Jacobs et al. 2023). By splitting the attention heads and forming complete Query (Q), Key (K) and Value (V) sequences, Ulysses parallelizes attention calculation across multiple GPUs. Its primary advantage lies in its communication pattern, which involves three All-to-All operations before attention computation and one afterward, resulting in relatively low communication overhead. However, its scalability is limited by the number of attention heads, and existing implementations do not overlap computation and communication, leaving GPU resources underutilized. The second is Ring-Attention (Li et al. 2021), which performs local attention computations on a partial sequence, and then gathers the K and V tensors across all devices using Peer-to-Peer (P2P) communication to complete global attention. This method supports much higher degrees of parallelism, but the increased communication overhead can negate its benefits, making Ulysses generally preferable when parallelism is not inherently restricted. Some prior works have combined both methods into Unified Ulysses-Ring SP (USP) (Fang and Zhao 2024; Fang et al. 2024a), mitigating the limited parallelism of Ulysses at the cost of additional communication overhead.

[Figure 1: High-level text-to-video pipeline diagram. A prompt is encoded by a text encoder; a pure-noise latent is refined by the diffusion backbone across multiple denoising steps into a clean latent; the clean latent is decoded/upscaled by a VAE decoder into the output video. The diagram groups operations into a “Denoising Stage” and a “Decoding Stage”.]

Figure 1: Text-to-video generation starts with encoding the input text and a pure noise latent into a semantic representation, which guides a diffusion model to iteratively refine a latent. The refined latent is then upsampled by a VAE decoder to generate the final video.

[Figure 2: Two sets of bar charts showing latency (seconds) and peak GPU memory usage (GB) for pipeline components (Text Encoder, Diffusion, VAE Decoder). (a) OpenSoraPlan model at 480×352×65 and 50 timesteps. (b) HunyuanVideo model at 256×128×33 and 50 timesteps. The figure contrasts diffusion as the time bottleneck and VAE decoding as the memory bottleneck.]

Figure 2: Latency and peak GPU memory usage of each component during inference (using eight GPUs with SP) for a single prompt in (a) OpenSoraPlan (PKU-YuanGroup 2025) model with a resolution of 480×352×65 and 50 timesteps (b) HunyuanVideo (Kong et al. 2024) model with a resolution of 256×128×33 and 50 timesteps.

While current system-level optimizations aim to speed up DiT-based video generation inference reserving the video quality, they still face two key issues: 1) communication between GPUs during the denoising phase often hampers inference efficiency, and 2) using a decoding VAE can quickly result in out-of-memory (OOM) errors, rendering decoding inefficient. In this paper, we introduce PipeDiT, a system-level optimized inference framework, which employs three innovative methods to reduce video generation latency while maintaining video output quality. First, we design a pipeline algorithm, named PipeSP, that overlaps communication and computation within Ulysses to hide some communication overhead and improve GPU utilization. Second, to address the GPU memory explosion in the decoding stage caused by colocation, we propose DeDiVAE to decouple the diffusion module and VAE decoder and onto two GPU groups. DeDiVAE greatly reduces peak GPU memory usage while allowing pipelined execution of decoding and denoising computations. Third, to address the suboptimal utilization of some GPUs caused by DeDiVAE, we further propose an attention co-processing (Aco) module which breaks down DiT into its linear-layer and attention-computation components. This fine-grained breakdown allows attention computation to proceed concurrently across both GPU groups in DeDiVAE, thereby improving GPU utilization. The main contributions of this paper are summarized as follows:

- We analyze the computation and communication patterns of Ulysses and propose an optimized version by pipelining communication and computation tasks in the denoising stage.
- To tackle GPU memory limitations during video generation and inefficiency caused by offloading, we propose a module-level pipeline parallelism that separates diffusion denoising and VAE decoding across different GPUs, significantly reducing peak memory consumption and improving the generation efficiency.
- To enhance GPU utilization in the decoupled setup, we introduce a fine-grained decoupling strategy that further decouples DiTs into its linear-layer and attention-computation components, allowing attention operations to be distributed across all GPUs.
- Building upon OpenSoraPlan (PKU-YuanGroup 2025) and HunyuanVideo (Tencent-Hunyuan 2025), we evaluate PipeDiT by measuring single-timestep runtime and multi-prompt inference latency under various configurations on two 8-GPU systems. The experimental results demonstrate the effectiveness and scalability of our optimizations.

## 2 Preliminaries & Motivations

Diffusion-based video generation comprises two stages as shown in Fig. 1: a Diffusion-based denoising stage that refines a latent representation over multiple timesteps and a variational autoencoder (VAE)-based decoding stage that upsamples the latent into a full-resolution video. The denoising stage is computationally heavy due to attention operations, while the decoding stage is highly memory-intensive due to upsampling to target resolution and frame rate.

We conduct a preliminary benchmark using two state-of-the-art video generation frameworks, OpenSoraPlan (PKU-YuanGroup 2025) and HunyuanVideo (Tencent-Hunyuan 2025) as shown in Fig. 2. This indicates that the diffusion stage takes significantly longer than the other two stages, but its memory usage is relatively small. Without offloading enabled, the VAE Decoder peaks at 44GB of memory when decoding the 256×128×33 latent, which is largely because the model parameters occupy a substantial amount of GPU memory. It is evident that during the entire diffusion-based video generation process, the denoising of the latent becomes the time bottleneck, while the decoding of the latent is the memory bottleneck.

In current mainstream video generation models, the diffusion backbone and the VAE decoder are typically colocated, which leads to serialized execution of diffusion computation and VAE upsampling. Without employing offloading or other memory-saving techniques, this design results in significant additional and inefficient memory consumption. Therefore, colocating the diffusion model and the VAE decoder is unfavorable for parallel video generation and hinders the generation of higher-resolution videos. The experimental results show that under the single-GPU memory constraint of 48 GB, OpenSoraPlan is unable to generate videos with resolutions larger than 1024×576×97 without offloading. Due to its larger model weights, HunyuanVideo cannot generate videos beyond 256×128×33 in resolution (see the experimental section for details).

Offloading is a commonly used strategy to reduce GPU memory consumption during inference (Abul-Fazl, Dina, and Fairuza 2025; Chen et al. 2024b). This strategy saves GPU memory by dynamically transferring model weights between the CPU and GPU. The primary advantage of this strategy is its implementation simplicity and its effectiveness in enabling higher-resolution video generation with limited GPU memory. Accordingly, offloading is adopted by several large-scale video generation systems—such as HunyuanVideo (Tencent-Hunyuan 2025), Wan (Wan et al. 2025), and OpenSoraPlan (PKU-YuanGroup 2025). However, offloading introduces significant CPU-GPU data transfer overhead, which depends on model size and bandwidth. The offloading overhead may easily dominate and hurt efficiency, while it cannot run the inference without offloading due to the high memory consumption of video generation.

Sequence parallelism (SP) (Li et al. 2023a; Wang et al. 2025; Wu et al. 2024) like Ulysses is a technique used to accelerate the processing of long input sequences on multiple GPUs. Ulysses achieves parallel computation by splitting along the attention head dimension, with different GPUs processing different attention heads. The computation model of Ulysses is illustrated in Fig. 3(a). After each GPU computes its portion of the sub-sequence’s Q, K, and V, three rounds of All-to-All communication are used to distribute Q, K, and V along the attention head dimension. The GPUs then concatenate the gathered Q, K, and V to form a complete sequence, but with only partial attention heads. After all attention heads have been computed, a final round of All-to-All communication is used to disperse the results along the sequence dimension and collect the data along the attention head dimension, resulting in a hidden state with partial sequence length but full attention heads. However, in the original Ulysses kernel, as illustrated in Fig. 3(a), a single All-to-All operation is issued only after all attention heads have been computed; during the waiting for this communication, GPUs remain idle, resulting in a waste of computational resources.

## 3 Methodology

### Pipelining Computation and Communication in SP

To address the issue of serial communication and computation in Ulysses, we propose a pipelined SP (PipeSP) algorithm that partitions the computation of attention along the head dimension and issues an All-to-All immediately after each head is processed, as illustrated in Fig. 3(b), thus overlapping communication with computation to improve the computation efficiency. Specifically, in the attention layer that has n heads, each head is processed independently. Thus, the n heads can be partitioned into n independent attention operations, each of which has only one head. After each head has been computed at its attention, its result can be communicated with other GPUs via an All-to-All operation. Thus, the operations of attention (computation) and All-to-All (communication) form a pipeline, which keeps GPU resources be fully utilized during the inference process. After the results of all heads have been gathered, a layout transformation is performed to align the result be identical with that without pipelining.

[Figure 3: Two diagrams. (a) Ulysses executes attention-head computations and communication sequentially; communication occurs after all heads complete, leaving idle gaps. (b) PipeSP pipelines “attention” and “All-to-All” per-head, then applies a post-processing “view → permute → view” to fix a head-order misalignment introduced by per-head pipelining.]

Figure 3: (a) The execution process of Ulysses, where computation and communication are executed sequentially. (b) Our optimized SP (PipeSP) by pipelining communication and computation. The subsequent post-processing resolves the misalignment issue introduced by the pipelining.

Algorithm 1: PipeSP: Overlapping Computation and Communication in SP

```text
Require: Q ∈ R^{B×h×S×D}, K, V, attention_mask

1: Initialize chunks, results, event_lst
2: for j ← 0 to h − 1 do
3:     result ← attention(Q[:, j, :, :], K[:, j, :, :], V[:, j, :, :], attention_mask[:, j, :, :])
4:     Append result to results
5:     Record event event_lst[j]
6:     Wait on CUDA stream for event_lst[j]
7:     hidden_states ← All-to-All(results[j])
8:     Append hidden_states to chunks
9: end for
10: hidden_states ← concat(chunks, dim = 1)
11: hidden_states ← view(−1, h, n, D)
12: hidden_states ← permute(0, 2, 1, 3)
13: hidden_states ← view(−1, h × n, D)
```

The PipeSP algorithm is shown in Algorithm 1. In lines 1–9, the Q, K, and V tensors are partitioned along the attention head dimension, and for each head, attention is computed over the full sequence. An event is recorded to mark the completion of this computation, and once all GPUs have completed the corresponding event, an All-to-All is triggered. In this step, each GPU receives the portion of the attention output corresponding to its local sequence slice for a single head. Lines 10–13 perform post-processing to reorder the collected results. This reordering is necessary because the optimized method collects attention outputs one head at a time, whereas the original method gathered them in a total different way, resulting in a misalignment shown in Fig. 3(b). To resolve this, the tensor must be reshaped and permuted using the sequence view → permute → view. Specifically, view(-1, h, n, D) reshapes the head dimension into a 2D layout of [h, n], permute(0, 2, 1, 3) swaps the GPU and head axes, and the final view(-1, nh, D) restores the expected layout. Mathematically, this process ensures the final tensor matches the original layout expected by the attention module, while enabling efficient communication–computation overlap. A formal proof of the correctness of the view–permute–view transformation is provided in the Supplementary Material.

### Memory-Efficient Diffusion–VAE Decoupling

To address the issues of low computational efficiency and poor GPU memory utilization caused by colocating the diffusion model and the VAE decoder, we propose Diffusion–VAE Module Decoupling (DeDiVAE) by breaking down the Diffusion module and the VAE module to two disjoint GPU groups: Denoising Group and Decoding Group. Specifically, for a given N-GPU system for video generation with DiT, DeDiVAE splits the N GPUs to Ndenoise GPUs as the Denoising Group and the other Ndecode = N − Ndenoise GPUs as the Decoding Group. Accordingly, full video generation model is split into the Diffusion backbone stored in the Denoising Group, and the VAE decoder stored in the Decoding Group. The decoupling effectively avoids the OOM problem for large models and high-resolution video generation. The latent outputs of the Denoising Group should be sent to the Decoding Group to generate the video.

At first glance, DeDiVAE might also lead to idle periods due to the data dependency between the two groups, potentially affecting inference efficiency. This issue can be addressed by implementing a pipeline execution with multiple prompts. Given that a video generation service typically handles multiple ongoing queries, multiple prompts (or queries) can be pipelined within our decoupled structure as demonstrated in Fig. 4. The decoding execution with VAE of the first prompt can be overlapped with the denoising execution with diffusion of the second prompt, which allows both GPU groups to keep busy. To maximize the utilization of GPU resources, we provide an effective analysis on how many GPUs should be assigned to the two groups.

Optimal GPU partitioning. Given N GPUs for inference, there are Ndenoise GPUs in the Denoising Group and Ndecode GPUs in the Decoding Group. Let Tdenoise denote the time of denoising one prompt on a single GPU and Tdecode denote the time of decoding one latent on a single GPU. During inference we accelerate the denoising with SP across Ndenoise GPUs, while the decoder uses data parallelism across Ndecode GPUs. Since the total workloads of inference are unchanged, to enable an maximal overlap is to make the execution time of the groups be identical. Thus, the first-order balance condition is

$$\left(\frac{T_{\text{denoise}}}{N_{\text{denoise}}} + T_{\text{comm}}\right) N_{\text{decode}} \approx T_{\text{decode}},$$

which yields the optimal $N_{\text{decode}}$ that maximizes GPU utilization:

$$N_{\text{decode}} \approx \left(\frac{T_{\text{decode}}}{T_{\text{decode}} + T_{\text{denoise}}}\right) N.$$

This allocation makes both stages finish a micro-batch in approximately the same time, preventing either group of GPUs from idling while the other is still computing. Since intra-node GPU communication is very fast, and PipeSP overlaps communication with computation to hide most of the communication overhead, omitting $T_{\text{comm}}$ does not affect the resulting resource allocation.

In practice, though we assign the GPUs in a balance way, the execution time of the diffusion stage may still dominate the overall execution time. To improve the efficiency, we design a new co-processing approach in DeDiVAE as introduced in the following section.

### Attention Co-processing

When denoising process is much slower than a VAE decoding, the Decoding GPUs idle during most of the generation window, and the pipeline cannot achieve a full overlap. Therefore, we propose Attention Co-processing (Aco) to utilize the idle time of the Decoding Group. We further split the DiT block into two disjoint kernels:

- Linear projections: $Q = XW_Q$, $K = XW_K$, $V = XW_V$,
- Attention kernel: $\mathrm{Attn}(Q, K, V)$,

and assign them to the two GPU groups. The Denoising GPUs keep the DiT weights and compute the linear projections; immediately afterwards they transmit the resulting Q, K, V tensors via point-to-point links to the Decoding GPUs, when Decoding GPUs are not decoding latents. Because the attention kernel depends only on Q, K, V, and the computations of different attention heads are independent in multi-head attention, the Decoding GPUs can execute it autonomously.

[Figure 4: Multi-prompt pipeline with two GPU groups. “Prompt 1” stage: Decoding group idle, so it assists attention by receiving Q/K/V via P2P send/recv and computing attention in parallel with the denoising group; results are aggregated via intra-group All-to-All and inter-group P2P. “Prompt 2” stage: Decoding group is busy running the VAE decoder on queued latents while the denoising group performs attention independently. Diagram includes dataflow labels (QKV, Q′K′V′, H hidden states, L latents), All-to-All, All-gather, and a note about “Padding or switch to Ring-Attention” when head count divisibility constraints apply.]

Figure 4: In the prompt 1 stage, the Denoising GPUs transmit the computed Q, K, and V tensors to the Decoding GPUs, enabling parallel attention computation across both groups. In the prompt 2 stage, the Denoising GPUs perform attention computation independently, while the Decoding GPUs execute decoding in parallel.

In the prompt 2 stage, as the decoding queue becomes non-empty, the Decoding GPUs are occupied with decoding latents from the queue. Consequently, the Denoising GPUs must perform attention computation independently. During this stage, denoising and decoding proceed in parallel.

It is worth noting that if the number of attention heads is not divisible by the number of Denoising GPUs, there are two possible strategies to handle this. For models that only adopt Ulysses, such as OpenSoraPlan (PKU-YuanGroup 2025), head dimension padding must be introduced to ensure balanced workload distribution. In contrast, models like HunyuanVideo (Tencent-Hunyuan 2025) and Wan (Wan et al. 2025) adopt Unified Sequence Parallelism (USP) (Fang and Zhao 2024), which allows flexible configuration of both the Ulysses degree and the Ring-Attention degree. When the number of heads is not divisible, we can switch Denoising GPUs from Ulysses to Ring-Attention, effectively changing the parallelism from head-wise splitting to sequence-wise splitting, thereby avoiding the overhead of padding and improving GPU utilization.

Performance Analysis. Let $t_L$ and $t_A$ denote the wall-clock time of one linear-projection and one attention kernel on Denoising GPUs, respectively, and let $N_{\text{denoise}}$ and $N_{\text{decode}}$ be the two GPU group sizes ($N_{\text{denoise}} + N_{\text{decode}} = N$). In the baseline decoupling only the denoising group participates:

$$T_{\text{baseline}} = t_L + t_A. \tag{1}$$

With Aco, the linear part still costs $t_L$, but the attention time scales inversely with the total number of GPUs that now share the work:

$$T_{\text{coop}} = t_L + t_A \cdot \frac{N_{\text{denoise}}}{N_{\text{denoise}} + N_{\text{decode}}}. \tag{2}$$

Hence the theoretical speed-up is

$$S = \frac{T_{\text{baseline}}}{T_{\text{coop}}} = \frac{t_L + t_A}{t_L + t_A \cdot \frac{N_{\text{denoise}}}{N}}. \tag{3}$$

Note that the above analysis assumes the number of attention heads $H$ is divisible by the number of Denoising GPUs $N_{\text{denoise}}$. If not divisible, and switching between Ulysses and Ring-Attention as in HunyuanVideo is not supported, then padding is required to balance the workload, leading to wasted GPU resources. For example, if $H = 24$ and 7 GPUs are used for denoising, padding increases the head count to 28 so that each GPU handles 4 heads. However, only 6 GPUs are effectively needed, and one GPU performs redundant computations. Our Attention Co-processing solves this issue by avoiding padding and ensuring all GPUs perform meaningful work, even when $H$ is not divisible by $N_{\text{denoise}}$.

## 4 Evaluation

### Experimental Setups

Baselines. We implement our PipeDiT on two state-of-the-art open-source video generation systems OpenSoraPlan at v1.3.0 (1) and HunyuanVideo (2). As the generation algorithm of PipeDiT is identical with OpenSoraPlan and HunyuanVideo, the generated videos are identical, so we mainly compare the time and memory efficiency. Note that OpenSoraPlan uses Ulysses, while HunyuanVideo integrates USP in xDiT (Fang et al. 2024a). The model sizes for OpenSoraPlan and HunyuanVideo are 2B and 13B parameters, respectively.

Performance Metrics. The performance metrics are video generation efficiency and GPU memory consumption. The efficiency metrics consist of two aspects. The first is the latency per timestep, which measures the optimization effect of PipeSP. The second metric is the overall latency for generating multiple videos from consecutive prompts, which measures the overall optimization effect of PipeDiT.

1. https://github.com/PKU-YuanGroup/Open-Sora-Plan
2. https://github.com/Tencent-Hunyuan/HunyuanVideo

### End-to-End Performance

Table 1: Latency and speedup for generating 10 videos with the baseline system and our optimized PipeDiT. Bold numbers indicate the results obtained with PipeDiT w/ Aco.

```text
Resolution
OpenSoraPlan (A6000)                         OpenSoraPlan (L40)
10                30                 50      10                30                 50
base opt spd↑      base opt spd↑      base opt spd↑            base opt spd↑      base opt spd↑      base opt spd↑
480 × 352 × 97     227 107 2.12×      420 304 1.38×      622 502 1.24×            252 154 1.64×      492 407 1.21×      738 657 1.12×
640 × 352 × 97     257 135 1.90×      522 389 1.34×      786 643 1.22×            303 206 1.47×      650 545 1.19×      983 883 1.11×
800 × 592 × 97     520 397 1.31×     1257 1097 1.15×    1994 1766 1.13×           646 517 1.25×     1609 1441 1.12×    2570 2373 1.08×
1024 × 576 × 97    555 430 1.29×     1360 1144 1.19×    2162 1832 1.18×           731 591 1.24×     1836 1639 1.12×    2940 2689 1.09×

Resolution
HunyuanVideo (A6000)                         HunyuanVideo (L40)
10                30                 50      10                30                 50
base opt spd↑      base opt spd↑      base opt spd↑            base opt spd↑      base opt spd↑      base opt spd↑
480 × 352 × 97     540 165 3.27×      767 445 1.72×      965 726 1.33×            676 229 2.95×      992 649 1.53×      1350 1068 1.26×
640 × 352 × 97     593 191 3.10×      865 531 1.63×     1142 907 1.26×            760 295 2.58×     1231 843 1.46×      1702 1392 1.22×
800 × 592 × 97    1082 506 2.14×     1880 1492 1.26×    2686 2470 1.09×          1694 923 1.84×     3291 2702 1.22×     4898 4482 1.09×
1024 × 576 × 97   1399 729 1.92×     2545 2090 1.22×    3726 3453 1.08×          2237 1333 1.68×    4576 3894 1.18×     6952 6453 1.08×
```

Table 2: Improvement in per-timestep latency with our PipeSP.

```text
OpenSoraPlan (A6000)
                480x352x65 480x352x129 640x352x65 640x352x129 800x592x65 800x592x129 1024x576x65 1024x576x129
Baseline (s)      1.67        1.30       1.21        2.10       2.21        4.98        2.57        6.86
PipeSP (s)        1.73        1.20       1.69        1.83       2.05        4.74        2.41        6.54
Speedup          0.97×       1.08×      0.72×       1.15×      1.08×       1.05×       1.07×       1.05×

OpenSoraPlan (L40)
                480x352x65 480x352x129 640x352x65 640x352x129 800x592x65 800x592x129 1024x576x65 1024x576x129
Baseline (s)      1.36        1.66       1.05        2.44       2.87        7.34        3.61        10.30
PipeSP (s)        1.42        1.57       1.01        2.34       2.74        7.05        3.44        9.95
Speedup          0.96×       1.06×      1.04×       1.04×      1.05×       1.04×       1.05×       1.04×
```

Testbeds. All experiments are conducted on two 8-GPU systems: 1) eight NVIDIA RTX A6000 48GB GPUs and 2) eight NVIDIA L40 48GB GPUs. More environment information can be found in Supplementary Material.

We configure different video resolutions, commonly used diffusion timesteps (10, 30, and 50), and 10 prompts (3) to compare generation latency as shown in Table 1, where “base” indicates the baseline using offloading, “opt” indicates our PipeDiT, and “spd” indicates the speedup of PipeDiT over the baseline. Since Aco does not always achieve improvement especially on low workload video generation, bold numbers indicate the results generated using PipeDiT w/ Aco.

From the results in Table 1, our PipeDiT always achieves faster inference speed by 1.08×-3.27× than baseline both OpenSoraPlan and HunyuanVideo. Particularly, PipeDiT yields the most notable speedups under lower resolutions, fewer frames, and shorter timesteps, reaching up to 3.27× (up to 4.02× as shown in Supplementary Material). As the resolution, frame count, and timesteps increase, the benefit diminishes since the computation time dominates and the relative impact of data transfer decreases, making offloading less of a bottleneck. Despite this, our PipeDiT on OpenSoraPlan with the A6000 platform still achieves 1.18× improvement over the baseline in the highest setting.

For different models, HunyuanVideo has more parameters, so its offloading takes longer time, making PipeDiT more effective to HunyuanVideo under lower resolutions and timesteps. In contrast, under higher resolutions and timesteps, the optimizations of PipeDiT in HunyuanVideo bring greater gains to OpenSoraPlan due to its shorter computation time. For different hardware, because the A6000 GPUs are connected with NVLink which delivers higher communication speed than L40. Thus, PipeDiT yields shorter per-timestep computation times compared to L40, and thus it has higher improvement on A6000 than L40.

3. Due to the page limit, we put the results with more comprehensive configurations in Supplementary Material.

### Effectiveness of PipeSP

To evaluate the effectiveness of PipeSP, we use eight representative resolution and frame configurations and measure the per-timestep latency, as shown in Table 2. Notably, PipeSP achieves a 15% performance improvement under the 640×352×129 configuration. The results also indicate that the optimization achieves the best performance under moderate resolutions, as it strikes a balance between overly short and excessively long computation times. When the resolution is low, the computation time is short, and the communication overhead introduced by overlap can offset its benefits. Conversely, at high resolutions, the proportion of communication time becomes less significant relative to the overall computation time, thereby reducing the optimization gains.

### Memory Efficiency of DeDiVAE

For the memory optimization comparison, we compare our DeDiVAE with the original implementation w/o offloading as the baseline. The second row in Table 4 presents the original implementation with offloading, while the third row shows the results of our DeDiVAE approach. As shown in the table, both our method and the offloading strategy significantly reduce memory consumption during inference. The baseline implementation fails with an OOM error under the highest resolution setting, indicating that without any memory optimization strategies, the model is unable to generate videos beyond 1024 × 576 × 129 in OpenSoraPlan and 480 × 352 × 129 in HunyuanVideo.

Table 4: Peak GPU memory usage (GB) and reduction ratio.

```text
OpenSoraPlan
Methods     480x352x129           640x352x129           800x592x129          1024x576x129
            Mem      ↓%           Mem      ↓%           Mem      ↓%          Mem      ↓%
Baseline    26.5      –           29.4      –           39.8      –          OOM       –
Offloading  18.4    30.6%         18.4    37.4%         19.1   52.0%         28.3   41.0%
DeDiVAE     18.0    32.1%         18.2    38.1%         18.6   53.3%         28.1   41.5%

HunyuanVideo
Methods     480x352x129           640x352x129           800x592x129          1024x576x129
            Mem      ↓%           Mem      ↓%           Mem      ↓%          Mem      ↓%
Baseline    OOM       –           OOM       –           OOM       –          OOM       –
Offloading  29.37   38.8%         29.38   38.8%         32.97  31.3%         33.01  31.2%
DeDiVAE     41.44   13.6%         41.43   13.7%         41.45  13.6%         42.12  12.2%
```

It should be noted that in HunyuanVideo, our DeDiVAE demonstrates greater peak memory consumption than the offloading method. This occurs because we colocated the text encoder with the VAE decoder. Given the substantial size of the text encoder in HunyuanVideo, colocating it with the DiT module, as done in OpenSoraPlan, is not feasible. This demonstrates that our method allows adaptable management of module positioning based on the attributes of the model.

[Figure 5: Heatmap comparing latency differences between PipeDiT without Aco and PipeDiT with Aco across multiple resolution/frame configurations. Cells indicate where Aco provides larger benefit, especially at higher workloads.]

Figure 5: The heatmap of the latency difference between the two methods: (1) PipeDiT w/o Aco and (2) PipeDiT w/ Aco.

### Ablation Study

To evaluate the performance improvements of various optimization methods, we fixed the number of timesteps to 30 and selected eight different resolutions and frame settings for ablation studies. The results are shown in Table 3, where “A” indicates the baseline offloading method, “B” indicates DeDiVAE, “C” indicates PipeSP, and “D” refers to Aco.

Table 3: Efficiency improvement of different optimization methods.

```text
OpenSoraPlan (A6000)
A B C D   480×352×65   480×352×129  640×352×65   640×352×129  800×592×65   800×592×129  1024×576×65  1024×576×129
          T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑
✓ ✗ ✗ ✗   314  1×     529  1×      368  1×      665  1×      777  1×      1875 1×      851  1×      1995 1×
✗ ✓ ✗ ✗   217  1.45×  452  1.17×   234  1.57×   500  1.33×   649  1.20×   1872 1.00×   702  1.21×   2138 0.93×
✗ ✓ ✓ ✗   200  1.57×  390  1.36×   250  1.47×   509  1.31×   649  1.20×   1847 1.02×   717  1.19×   1936 1.03×
✗ ✓ ✓ ✓   261  1.20×  414  1.28×   296  1.24×   507  1.31×   645  1.20×   1652 1.14×   683  1.25×   1690 1.18×

HunyuanVideo (A6000)
A B C D   480×352×65   480×352×129  640×352×65   640×352×129  800×592×65   800×592×129  1024×576×65  1024×576×129
          T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑   T(s)↓ Spd↑
✓ ✗ ✗ ✗   636  1×     911  1×      695  1×      1104 1×      1294 1×      2681 1×      1676 1×      3733 1×
✗ ✓ ✗ ✗   340  1.87×  681  1.34×   403  1.72×   824  1.34×   984  1.32×   2501 1.07×   1374 1.22×   3680 1.01×
✗ ✓ ✓ ✗   345  1.84×  701  1.30×   404  1.72×   824  1.34×   983  1.32×   2499 1.07×   1374 1.22×   3675 1.02×
✗ ✓ ✓ ✓   327  1.94×  595  1.53×   396  1.76×   741  1.49×   942  1.37×   2259 1.19×   1242 1.35×   3090 1.21×
```

The results show that PipeSP demonstrates significant performance improvements on OpenSoraPlan. This is primarily because OpenSoraPlan incurs longer All-to-All communication times, and due to its non-modular design, we were able to partially overlap the computation of Q, K, and V with the three All-to-All communications that occur before the attention computation.

For DeDiVAE, it achieves substantial efficiency gains under lower resolutions. However, as the resolution increases, the drawback of having fewer GPUs allocated to denoising becomes more apparent. Introducing Aco helps to address this limitation. Even under the highest resolution setting, the combined approach still delivers considerable performance benefits. The performance difference of PipeDiT w/ Aco and w/o Aco is shown in Fig. 5, which indicates that Aco improves performance on high workload tasks.

## 5 Conclusion

In this study, we proposed a system-level optimization system named PipeDiT for accelerating the video generation with diffusion transformer (DiT) based models. There are three key innovations in PipeDiT: 1) a pipelining algorithm named PipeSP for sequence parallelism (SP) for enable overlapping between communication and computation tasks, 2) a module decoupling method named DeDiVAE by breaking down the diffusion module and VAE module to two GPU groups to reduce the memory consumption, and 3) an attention co-processing approach named Aco, which leverages the idle decoding GPU group to assist with denoising module execution. Our PipeDiT is implemented atop two state-of-the-art video generation frameworks OpenSoraPlan and HunyuanVideo. Extensive experiments were conducted on two GPU systems and the results indicate that our PipeDiT not only significantly reduces memory consumption but also greatly enhances the generation efficiency.

## Acknowledgments

The research was supported in part by the National Natural Science Foundation of China (NSFC) under Grant No. 62302123, and the Shenzhen Science and Technology Program under Grant No. KJZD20240903104103005, KJZD20230923115113026, and KQTD20240729102154066.

## References

Abul-Fazl, S.; Dina, R.; and Fairuza, H. 2025. Diffusion Models at Scale: Techniques, Applications, and Challenges.

Blattmann, A.; Dockhorn, T.; Kulal, S.; Mendelevitch, D.; Kilian, M.; Lorenz, D.; Levi, Y.; English, Z.; Voleti, V.; Letts, A.; et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127.

Chen, J.; Ge, C.; Xie, E.; Wu, Y.; Yao, L.; Ren, X.; Wang, Z.; Luo, P.; Lu, H.; and Li, Z. 2024a. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, 74–91. Springer.

Chen, M.; Mei, S.; Fan, J.; and Wang, M. 2024b. Opportunities and challenges of diffusion models for generative AI. National Science Review, 11(12): nwae348.

Chen, P.; Shen, M.; Ye, P.; Cao, J.; Tu, C.; Bouganis, C.-S.; Zhao, Y.; and Chen, T. 2024c. Delta-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers. arXiv preprint arXiv:2406.01125.

Cho, J.; Puspitasari, F. D.; Zheng, S.; Zheng, J.; Lee, L.-H.; Kim, T.-H.; Hong, C. S.; and Zhang, C. 2024. Sora as an agi world model? a complete survey on text-to-video generation. arXiv preprint arXiv:2403.05131.

Fan, W.; Si, C.; Song, J.; Yang, Z.; He, Y.; Zhuo, L.; Huang, Z.; Dong, Z.; He, J.; Pan, D.; et al. 2025. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453.

Fang, J.; Pan, J.; Sun, X.; Li, A.; and Wang, J. 2024a. xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism. arXiv preprint arXiv:2411.01738.

Fang, J.; Pan, J.; Wang, J.; Li, A.; and Sun, X. 2024b. Pipefusion: Patch-level pipeline parallelism for diffusion transformers inference. arXiv preprint arXiv:2405.14430.

Fang, J.; and Zhao, S. 2024. Usp: A unified sequence parallelism approach for long context generative ai. arXiv preprint arXiv:2405.07719.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33: 6840–6851.

Jacobs, S. A.; Tanaka, M.; Zhang, C.; Zhang, M.; Song, S. L.; Rajbhandari, S.; and He, Y. 2023. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509.

Kong, W.; Tian, Q.; Zhang, Z.; Min, R.; Dai, Z.; Zhou, J.; Xiong, J.; Li, X.; Wu, B.; Zhang, J.; et al. 2024. Hunyuan-video: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603.

Li, C.; Huang, D.; Lu, Z.; Xiao, Y.; Pei, Q.; and Bai, L. 2024a. A survey on long video generation: Challenges, methods, and prospects. arXiv preprint arXiv:2403.16407.

Li, D.; Shao, R.; Xie, A.; Xing, E.; Gonzalez, J. E.; Stoica, I.; Ma, X.; and Zhang, H. 2023a. Lightseq: Sequence level parallelism for distributed training of long context transformers.

Li, M.; Cai, T.; Cao, J.; Zhang, Q.; Cai, H.; Bai, J.; Jia, Y.; Li, K.; and Han, S. 2024b. Distrifusion: Distributed parallel inference for high-resolution diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7183–7193.

Li, S.; Xue, F.; Baranwal, C.; Li, Y.; and You, Y. 2021. Sequence parallelism: Long sequence training from system perspective. arXiv preprint arXiv:2105.13120.

Li, Y.; Wang, H.; Jin, Q.; Hu, J.; Chemerys, P.; Fu, Y.; Wang, Y.; Tulyakov, S.; and Ren, J. 2023b. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36: 20662–20678.

Liu, F.; Zhang, S.; Wang, X.; Wei, Y.; Qiu, H.; Zhao, Y.; Zhang, Y.; Ye, Q.; and Wan, F. 2025. Timestep Embedding Tells: It’s Time to Cache for Video Diffusion Model. In Proceedings of the Computer Vision and Pattern Recognition Conference, 7353–7363.

Liu, S.; Yu, W.; Tan, Z.; and Wang, X. 2024. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097.

Luo, J.; Xiao, Y.; Xu, J.; You, Y.; Lu, R.; Tang, C.; Jiang, J.; and Wang, Z. 2025. Accelerating Parallel Diffusion Model Serving with Residual Compression. arXiv preprint arXiv:2507.17511.

Ma, X.; Fang, G.; and Wang, X. 2024. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 15762–15772.

Ma, X.; Wang, Y.; Jia, G.; Chen, X.; Liu, Z.; Li, Y.-F.; Chen, C.; and Qiao, Y. 2024. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048.

PKU-YuanGroup. 2025. Open-Sora-Plan. https://github.com/PKU-YuanGroup/Open-Sora-Plan. Accessed: 2025-07-19.

Selvaraju, P.; Ding, T.; Chen, T.; Zharkov, I.; and Liang, L. 2024. Fora: Fast-forward caching in diffusion transformer acceleration. arXiv preprint arXiv:2407.01425.

Shih, A.; Belkhale, S.; Ermon, S.; Sadigh, D.; and Anari, N. 2023. Parallel sampling of diffusion models. Advances in Neural Information Processing Systems, 36: 4263–4276.

Sohl-Dickstein, J.; Weiss, E.; Maheswaranathan, N.; and Ganguli, S. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, 2256–2265. pmlr.

Song, Y.; and Ermon, S. 2019. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32.

Sun, R.; Zhang, Y.; Shah, T.; Sun, J.; Zhang, S.; Li, W.; Duan, H.; Wei, B.; and Ranjan, R. 2024a. From sora what we can see: A survey of text-to-video generation. arXiv preprint arXiv:2405.10674.

Sun, W.; Qin, Z.; Li, D.; Shen, X.; Qiao, Y.; and Zhong, Y. 2024b. Linear attention sequence parallelism. arXiv preprint arXiv:2404.02882.

Tencent-Hunyuan. 2025. HunyuanVideo: A Systematic Framework for Large Video Generation Model. https://github.com/Tencent-Hunyuan/HunyuanVideo. Accessed: 2025-07-19.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

Wang, Y.; Wang, S.; Zhu, S.; Fu, F.; Liu, X.; Xiao, X.; Li, H.; Li, J.; Wu, F.; and Cui, B. 2025. Flexsp: Accelerating large language model training via flexible sequence parallelism. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, 421–436.

Wu, B.; Liu, S.; Zhong, Y.; Sun, P.; Liu, X.; and Jin, X. 2024. Loongserve: Efficiently serving long-context large language models with elastic sequence parallelism. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, 640–654.

Yang, Z.; Teng, J.; Zheng, W.; Ding, M.; Huang, S.; Xu, J.; Yang, Y.; Hong, W.; Zhang, X.; Feng, G.; et al. 2024. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072.

Zhang, X.; Luo, Z.; and Lu, M. E. 2024. Partially Conditioned Patch Parallelism for Accelerated Diffusion Model Inference. arXiv preprint arXiv:2412.02962.

Zhao, X.; Cheng, S.; Chen, C.; Zheng, Z.; Liu, Z.; Yang, Z.; and You, Y. 2024a. Dsp: Dynamic sequence parallelism for multi-dimensional transformers. arXiv preprint arXiv:2403.10266.

Zhao, X.; Jin, X.; Wang, K.; and You, Y. 2024b. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588.

## Supplementary Material

### Complete End-to-End Performance Results

In the main body, we present a subset of the End-to-End Performance Results, while Table 6 provides the complete results. We evaluate our model under multiple resolutions ranging from 480×352 to 1024×576, covering common video qualities such as 480p and 576p. The timestep values are arranged in descending order, where fewer timesteps lead to lower latency but also lower generation quality.

Results demonstrate that PipeDiT achieves notable acceleration across both models and platforms, proving its effectiveness for lightweight frameworks like OpenSoraPlan and large-scale systems like HunyuanVideo. PipeDiT is thus applicable to video generation models of various sizes.

Under low to medium resolutions and shorter timesteps, PipeDiT achieves up to 4.02× speedup. Even at the highest resolution and timestep, it still delivers 1.06× to 1.17× improvement. As the current trend in DiT-based inference optimization moves toward using fewer timesteps and more aggressively compressed VAE decoders to reduce denoising latency, the speedup brought by PipeDiT is expected to increase as the denoising time becomes smaller.

Additionally, some recent video generation models, such as Wan2.2, have adopted Mixture-of-Experts (MoE) architectures that significantly increase model size. In such cases, traditional offloading approaches become less viable, while PipeDiT offers a scalable and efficient alternative well-suited to these future scenarios.

Table 6: Latency and speedup for generating 10 videos with the baseline system and our optimized PipeDiT. Bold numbers indicate the results obtained with PipeDiT w/ Aco.

```text
Resolution
OpenSoraPlan (A6000) OpenSoraPlan (L40)
10 20 30 40 50 10 20 30 40 50
base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑
480×352×65 177 73 2.42× 240 138 1.74× 314 200 1.57× 379 264 1.44× 437 329 1.33× 204 105 1.94× 274 202 1.35× 355 298 1.19× 436 393 1.11× 512 481 1.06×
480×352×97 227 107 2.12× 321 205 1.57× 420 304 1.38× 521 403 1.29× 622 502 1.24× 252 154 1.64× 368 278 1.32× 492 407 1.21× 613 531 1.15× 738 657 1.12×
480×352×129 262 135 1.94× 400 263 1.52× 529 389 1.36× 666 518 1.28× 794 644 1.23× 310 200 1.55× 484 372 1.30× 652 544 1.20× 824 714 1.15× 996 882 1.13×
640×352×65 203 92 2.21× 285 170 1.68× 368 250 1.47× 491 332 1.48× 536 411 1.30× 224 129 1.74× 333 244 1.36× 425 348 1.22× 527 454 1.16× 638 562 1.14×
640×352×97 257 135 1.90× 388 262 1.48× 522 389 1.34× 653 521 1.25× 786 643 1.22× 303 206 1.47× 473 373 1.27× 650 545 1.19× 820 713 1.15× 983 883 1.11×
640×352×129 310 176 1.76× 492 342 1.44× 665 507 1.31× 835 660 1.27× 1007 811 1.24× 368 253 1.45× 590 471 1.25× 808 685 1.18× 1030 902 1.14× 1240 1118 1.11×
800×592×65 347 228 1.52× 571 436 1.31× 777 645 1.20× 999 841 1.19× 1205 1037 1.16× 410 296 1.39× 669 555 1.21× 935 815 1.15× 1201 1073 1.12× 1471 1331 1.11×
800×592×97 520 397 1.31× 899 758 1.19× 1257 1097 1.15× 1637 1433 1.14× 1994 1766 1.13× 646 517 1.25× 1124 980 1.15× 1609 1441 1.12× 2013 1903 1.06× 2570 2373 1.08×
800×592×129 751 621 1.21× 1316 1137 1.16× 1875 1651 1.14× 2447 2174 1.13× 3010 2689 1.12× 959 791 1.21× 1709 1515 1.13× 2464 2239 1.10× 3223 2964 1.09× 3977 3688 1.08×
1024×576×65 376 251 1.50× 614 479 1.28× 851 684 1.24× 1094 888 1.23× 1323 1089 1.21× 476 354 1.34× 801 661 1.21× 1114 970 1.15× 1440 1280 1.13× 1760 1588 1.11×
1024×576×97 555 430 1.29× 959 799 1.20× 1360 1144 1.19× 1762 1491 1.18× 2162 1832 1.18× 731 591 1.24× 1279 1115 1.15× 1836 1639 1.12× 2387 2164 1.10× 2940 2689 1.09×
1024×576×129 797 652 1.22× 1402 1176 1.19× 1995 1698 1.17× 2600 2206 1.18× 3194 2726 1.17× 1094 916 1.19× 1970 1751 1.13× 2848 2582 1.10× 3722 3416 1.09× 4599 4251 1.08×

Resolution
HunyuanVideo (A6000) HunyuanVideo (L40)
10 20 30 40 50 10 20 30 40 50
base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑ base opt spd↑
480×352×65 482 120 4.02× 563 219 2.57× 636 327 1.94× 702 431 1.63× 766 520 1.47× 517 157 3.29× 632 298 2.12× 764 437 1.75× 850 578 1.47× 983 718 1.37×
480×352×97 540 165 3.27× 641 306 2.09× 767 445 1.72× 850 585 1.45× 965 726 1.33× 676 229 2.95× 838 439 1.91× 992 649 1.53× 1163 859 1.35× 1350 1068 1.26×
480×352×129 630 206 3.06× 767 395 1.94× 911 595 1.53× 1071 791 1.35× 1201 985 1.22× 787 312 2.52× 1038 602 1.72× 1288 892 1.44× 1555 1182 1.32× 1804 1472 1.23×
640×352×65 518 142 3.65× 596 269 2.22× 695 396 1.76× 791 515 1.54× 882 639 1.38× 601 195 3.08× 749 376 1.99× 902 555 1.63× 1041 735 1.42× 1206 915 1.32×
640×352×97 593 191 3.10× 733 360 2.04× 865 531 1.63× 1008 701 1.44× 1142 907 1.26× 760 295 2.58× 1021 571 1.79× 1231 843 1.46× 1460 1119 1.30× 1702 1392 1.22×
640×352×129 710 262 2.71× 899 503 1.79× 1104 741 1.49× 1272 987 1.29× 1456 1228 1.19× 987 409 2.41× 1313 797 1.65× 1637 1187 1.38× 1954 1573 1.24× 2305 1961 1.18×
800×592×65 833 321 2.60× 1075 633 1.70× 1294 942 1.37× 1531 1252 1.22× 1792 1561 1.15× 1160 527 2.20× 1601 1028 1.56× 2018 1527 1.32× 2487 2027 1.23× 2921 2528 1.16×
800×592×97 1082 506 2.14× 1499 1000 1.50× 1880 1492 1.26× 2281 1985 1.15× 2686 2470 1.09× 1694 923 1.84× 2484 1812 1.37× 3291 2702 1.22× 4115 3592 1.15× 4898 4482 1.09×
800×592×129 1467 772 1.90× 2062 1514 1.36× 2681 2259 1.19× 3302 3006 1.10× 3920 3756 1.04× 2311 1374 1.68× 3511 2686 1.31× 4717 4001 1.18× 5926 5317 1.11× 7136 6632 1.08×
1024×576×65 997 430 2.32× 1333 835 1.60× 1676 1242 1.35× 1989 1645 1.21× 2324 2051 1.13× 1486 760 1.96× 2146 1490 1.44× 2817 2223 1.27× 3473 2952 1.18× 4123 3682 1.12×
1024×576×97 1399 729 1.92× 1987 1425 1.39× 2545 2090 1.22× 3150 2771 1.14× 3726 3453 1.08× 2237 1333 1.68× 3395 2612 1.30× 4576 3894 1.18× 5762 5173 1.11× 6952 6453 1.08×
1024×576×129 1918 1097 1.75× 2836 2078 1.36× 3733 3090 1.21× 4638 4206 1.10× 5545 5240 1.06× 3187 2024 1.57× 5017 3984 1.26× 6846 5942 1.15× 8658 7899 1.10× 10472 9856 1.06×
```

### Complete Results of Ablation Study

In the main text, we only present the ablation results on the A6000 platform. Table 7 shows the full ablation study results across all platforms, where “A” indicates the baseline offloading method, “B” indicates DeDiVAE, “C” indicates PipeSP, and “D” refers to Aco. We observe that PipeDiT achieves more significant performance improvements on the A6000 platform compared to the L40 platform. This is primarily because, under the same model and input configurations, the end-to-end latency on the A6000 is lower than that on the L40 for both the baseline and PipeDiT. As a result, the proportion of offloading time—eliminated by PipeDiT—accounts for a larger share of the total runtime on the A6000 system, leading to a higher relative speedup. Furthermore, the A6000’s support for high-bandwidth NVLink interconnects allows more efficient inter-GPU communication compared to the PCIe-based L40. This architectural advantage further amplifies the effectiveness of PipeDiT’s PipeSP. These results demonstrate that PipeDiT can adapt well across heterogeneous hardware configurations while achieving especially notable improvements on platforms equipped with high interconnect bandwidth and lower baseline latencies.

Table 7: Efficiency improvement of different optimization methods.

```text
OpenSoraPlan(A6000)
A B C D
480×352×65 480×352×129 640×352×65 640×352×129 800×592×65 800×592×129 1024×576×65 1024×576×129
T(s)↓ Spd↑ T(s)↓ Spd↑ T(s)↓ Spd↑ T(s)↓ Spd↑ T(s)↓ Spd↑ T(s)↓ Spd↑ T(s)↓ Spd↑ T(s)↓ Spd↑
✓ ✗ ✗ ✗ 314 1× 529 1× 368 1× 665 1× 777 1× 1875 1× 851 1× 1995 1×
✗ ✓ ✗ ✗ 217 1.45× 452 1.17× 234 1.57× 500 1.33× 649 1.20× 1872 1.00× 702 1.21× 2138 0.93×
✗ ✓ ✓ ✗ 200 1.57× 390 1.36× 250 1.47× 509 1.31× 649 1.20× 1847 1.02× 717 1.19× 1936 1.03×
✗ ✓ ✓ ✓ 261 1.20× 414 1.28× 296 1.24× 507 1.31× 645 1.20× 1652 1.14× 683 1.25× 1690 1.18×

OpenSoraPlan(L40)
✓ ✗ ✗ ✗ 355 1× 652 1× 425 1× 808 1× 935 1× 2464 1× 1114 1× 2848 1×
✗ ✓ ✗ ✗ 274 1.30× 609 1.07× 372 1.14× 792 1.02× 962 0.97× 2816 0.88× 1161 0.96× 3257 0.87×
✗ ✓ ✓ ✗ 299 1.19× 621 1.05× 372 1.14× 789 1.02× 963 0.97× 2819 0.87× 1158 0.96× 3231 0.88×
✗ ✓ ✓ ✓ 298 1.19× 544 1.20× 348 1.22× 685 1.18× 815 1.15× 2239 1.10× 970 1.15× 2582 1.10×

HunyuanVideo(A6000)
✓ ✗ ✗ ✗ 636 1× 911 1× 695 1× 1104 1× 1294 1× 2681 1× 1676 1× 3733 1×
✗ ✓ ✗ ✗ 340 1.87× 681 1.34× 403 1.72× 824 1.34× 984 1.32× 2501 1.07× 1374 1.22× 3680 1.01×
✗ ✓ ✓ ✗ 345 1.84× 701 1.30× 404 1.72× 824 1.34× 983 1.32× 2499 1.07× 1374 1.22× 3675 1.02×
✗ ✓ ✓ ✓ 327 1.94× 595 1.53× 396 1.76× 741 1.49× 942 1.37× 2259 1.19× 1242 1.35× 3090 1.21×

HunyuanVideo(L40)
✓ ✗ ✗ ✗ 764 1× 1288 1× 902 1× 1637 1× 2018 1× 4717 1× 2817 1× 6846 1×
✗ ✓ ✗ ✗ 466 1.64× 1087 1.18× 600 1.50× 1380 1.19× 1690 1.19× 4751 0.99× 2443 1.15× 7180 0.95×
✗ ✓ ✓ ✗ 468 1.63× 1086 1.19× 599 1.51× 1380 1.19× 1687 1.20× 4749 0.99× 2441 1.15× 7175 0.95×
✗ ✓ ✓ ✓ 437 1.75× 892 1.44× 555 1.63× 1187 1.38× 1527 1.32× 4001 1.18× 2223 1.27× 5942 1.15×
```

### Consistency Proof of PipeSP

As PipeSP generates misordered results which are then transformed to the original layout, we prove that the results of PipeSP is identical with the original SP.

Resulting mis-order After all heads have been processed, every GPU owns the hidden state of its sub-sequence, but the global head order is now interleaved: for GPU index $i \in \{0, \ldots, n-1\}$ and local head $j \in \{0, \ldots, h-1\}$,

$$k_{\text{orig}}(i, j) = ih + j,\quad k_{\text{mod}}(i, j) = jn + i. \tag{4}$$

The tensor $T_{\text{mod}} \in \mathbb{R}^{B \times H \times D}$ therefore has head $k_{\text{mod}}(i, j)$ where the original order expects $k_{\text{orig}}(i, j)$; a re-ordering is mandatory.

Let $B$ be the mini-batch, $n$ the number of GPUs, $h$ heads per GPU ($H = nh$), and $D$ the head dimension. Since the sequence dimension $S$ is not involved in the operations considered in this section, it is omitted for brevity. Define the reshape map $\phi_{h,n}: \mathbb{R}^{B \times H \times D} \to \mathbb{R}^{B \times h \times n \times D}$,

$$(\phi_{h,n} T)[b, j, i, d] = T[b, jn + i, d], \tag{5}$$

and its inverse $\phi^{-1}_{h,n}$ that merges $(i, j)$ back into a linear head index. Let $\pi: \mathbb{R}^{B \times h \times n \times D} \to \mathbb{R}^{B \times n \times h \times D}$,

$$(\pi X)[b, i, j, d] = X[b, j, i, d]. \tag{6}$$

The code sequence `view(-1, h, n, D) -> permute(0, 2, 1, 3) -> view(-1, nh, D)` implements the composite map

$$\Psi = \phi^{-1}_{h,n} \circ \pi \circ \phi_{h,n}: \mathbb{R}^{B \times H \times D} \to \mathbb{R}^{B \times H \times D}. \tag{7}$$

Alignment proof For any $b, d, i, j$ one has

$$
\begin{aligned}
(\Psi T_{\text{mod}})[b, k_{\text{orig}}(i, j), d]
&= (\phi^{-1}_{h,n}\, \pi\, \phi_{h,n}\, T_{\text{mod}})[b, ih + j, d] \\\\
&= (\pi\, \phi_{h,n}\, T_{\text{mod}})[b, i, j, d] \\\\
&= (\phi_{h,n}\, T_{\text{mod}})[b, j, i, d] \\\\
&= T_{\text{mod}}[b, jn + i, d] \\\\
&= T_{\text{mod}}[b, k_{\text{mod}}(i, j), d].
\end{aligned}
$$

Hence $\Psi T_{\text{mod}} = T_{\text{orig}}$: after the two view plus one permute, the interleaved tensor is mapped exactly onto the head-contiguous layout expected by the original Ulysses implementation.

### Experimental Environment

Our experiments are conducted entirely on A6000 and L40 GPUs. The hardware configurations of the test platforms are shown in Table 5.

Table 5: Testbed Hardware Specifications.

| Component | Specification (A6000 platform) | Specification (L40 platform) |
|---|---|---|
| CPU | Intel® Xeon® Platinum 8358, @2.60 GHz | Intel® Xeon® Platinum 8358, @2.60 GHz |
| GPU | 8× Nvidia RTX A6000, 48GB GDDR6 | 8× Nvidia L40, 48GB GDDR6 |
| NVLink | 112.5GB/s (4×) | - |
| PCIe | 4.0 (x16) | 4.0 (x16) |

### Consistency Proof of Generated Results

Since our optimization focuses solely on resource allocation and computational workload balancing, the algorithmic logic remains fully consistent with the original method. As a result, the generated outputs are identical to those of the original algorithm. Fig. 6 presents the outputs from both the original method and our optimized approach under the same prompt, experimental configuration, and sampled frame index. The results clearly demonstrate that the two generations are perfectly identical.

[Figure 6: Side-by-side qualitative comparison images. Left: “Open-Sora-Plan w/o PipeDiT”. Right: “Open-Sora-Plan w/ PipeDiT”. The figure shows matching generated frames, supporting that PipeDiT preserves output equivalence.]

Figure 6: The generation results show that the outputs produced by PipeDiT are consistent with those of the original algorithm.

