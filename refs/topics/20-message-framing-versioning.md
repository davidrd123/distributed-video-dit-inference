---
status: stub
---

# Topic 20: Message framing and versioning

When pipeline stages communicate over network boundaries (multi-node inference), you need message framing (length-prefixed or delimited) and versioning for forward/backward compatibility as the system evolves.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| message-framing | Message Framing | low | pending |
| framing-textbook | Framing (Computer Networks: A Systems Approach) | low | pending |
| protobuf-guide | Protocol Buffers Language Guide (proto3) | low | pending |

## Implementation context

Both TP and PP bringup converged on the same pattern: a tiny fixed header (`call_id`, `chunk_index`, epochs, `action`) that determines whether a payload follows, plus a versioned payload manifest (`tensor_specs`) that lets receivers allocate tensors safely. TP v0 uses a 5×int64 header (~40 bytes) and `TPAction.{NOOP,INFER,SHUTDOWN}`; PP contracts similarly use `PPAction` plus `pp_envelope_version=1` and now a globally monotonic `call_id` (replacing the earlier `chunk_id` field). The key guardrail is “validate/pickle/spec everything before sending the header” to avoid stranding the receiver mid-message.

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md` (TPControlHeader + tensor_specs), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (anti-stranding Step A1 + contract changes).

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/control.py` (TPControlHeader + tensor_specs framing)
- `scope-drd/src/scope/core/distributed/pp_contract.py` (`PPEnvelopeV1`/`PPResultV1` versioned schema)
- `scope-drd/src/scope/core/distributed/pp_control.py` (PP preflight + header-first send pattern)

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
