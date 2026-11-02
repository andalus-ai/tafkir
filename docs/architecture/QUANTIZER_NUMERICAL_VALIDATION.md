# Quantizer numerical validation status

This document distinguishes provider discovery from algorithm readiness. A provider
must not advertise support to Aljabr until its representation, metadata, and
round-trip behavior are validated against a trusted reference implementation.

## Current integration availability

| Provider | Current status | Reason / next acceptance gate |
|---|---|---|
| GPTQ | Disabled | The adapter previously returned zero-filled packed data and did not attach quantization metadata. Implement Hessian/calibration-aware quantization, canonical GPTQ packing, and reference-vector parity before enabling. |
| AWQ | Disabled | The adapter used simplified min/max rather than activation-aware scaling and did not fully validate packed layout/edge dimensions. Add calibration-driven scaling, canonical AutoAWQ-compatible serialization, and reference parity. |
| AutoRound | Disabled | The adapter used naive min/max rounding and its packed dimensions/metadata were not proven compatible with AutoRound. Implement the optimization loop and canonical format, then verify reference parity. |
| TurboQuant | Disabled | The adapter stored codebook indices as one byte per element without persisting all codebook/rotation/scale state needed for reliable decoding. Define a stable serialized representation and verify deterministic round trips. |
| BitsAndBytes NF4 | Experimental / available for NF4 only | The adapter currently advertises only 4-bit BNB/NF4. INT8 is deliberately not advertised. Validate codebook mapping, block absmax handling, odd element counts, partial blocks, and parity with bitsandbytes reference vectors before production claims. |

## Required validation contract

Every enabled quantizer must pass all of the following:

1. **Representation** — packed byte/word order, bit width, group/block size, padding, shape, dtype, scales, zero-points, codebook and auxiliary state are explicit and round-trip through serialization.
2. **Numerics** — deterministic fixed-vector comparisons against a trusted implementation; report max absolute error, RMSE, relative error where defined, and cosine similarity.
3. **Edge cases** — zero tensors, constant groups, negative-only and positive-only groups, values at clipping boundaries, non-multiple group sizes, odd element counts, non-contiguous input, and NaN/Infinity policy.
4. **Lifecycle** — metadata remains valid for the lifetime of the returned tensor; no closed FFM arena may back scales or zero-points.
5. **Selection** — supports returns true only for the exact strategy and bit-width implemented by that provider. Unsupported requests must fail before writing output.
6. **Performance** — benchmark throughput and peak memory separately from correctness; an optimization must not change reference results beyond the declared tolerance.

## Release policy

- Disabled providers must not be selected by QuantizerRegistry or QuantizationEngine.
- NF4 availability is experimental, not a production-quality claim.
- Enabling a provider requires committed reference fixtures, numerical tests, format documentation, and CI evidence on the supported JDK/runtime.
