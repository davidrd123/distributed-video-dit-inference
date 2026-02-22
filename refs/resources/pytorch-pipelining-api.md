# PyTorch torch.distributed.pipelining API

| Field | Value |
|-------|-------|
| Source | https://docs.pytorch.org/docs/stable/distributed.pipelining.html |
| Type | docs |
| Topics | 14 |
| Status | stub |

## Why it matters

Documents the PyTorch pipeline parallelism API including ScheduleGPipe, Schedule1F1B, ScheduleInterleaved1F1B, ScheduleInterleavedZeroBubble, and ScheduleZBVZeroBubble. The code is the best reference for how these schedules manage activation memory and inter-stage communication in practice.

## Key sections

<!-- Most relevant sections/pages for the project -->

## Core claims

1. **Claim**: ...
   **Evidence**: *pending extraction*

## API surface / configuration

<!-- Key APIs, env vars, config options relevant to the project -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [gpipe](gpipe.md) -- ScheduleGPipe implements this paper's approach
- [pipedream-2bw](pipedream-2bw.md) -- Schedule1F1B implements the 1F1B schedule
- [zero-bubble-pp](zero-bubble-pp.md) -- ScheduleInterleavedZeroBubble and ScheduleZBVZeroBubble implement zero-bubble
- [pipedit](pipedit.md) -- pipeline-parallel DiT inference that may use these schedules
- [funcol-rfc-93173](funcol-rfc-93173.md) -- functional collectives used for inter-stage communication
