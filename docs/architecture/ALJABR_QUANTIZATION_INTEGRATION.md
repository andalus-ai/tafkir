# Aljabr SafeTensors Quantization Integration

## Purpose

This document defines the Tafkir-owned integration boundary between Tafkir's quantization implementations and Aljabr's SafeTensors runtime.

The dependency direction is one-way:

```
Tafkir quantizer implementations
        |
        v
Tafkir Aljabr integration extension
        |
        v
Aljabr SafeTensors quantization SPI
```

Aljabr owns the tensor-level contract and runtime. Tafkir owns training- and adaptation-oriented quantization algorithms and the adapters that expose those algorithms to Aljabr. Aljabr production modules must not depend on Tafkir artifacts or packages.

## Current repository state

The repository already has an `extensions/aljabr/tafkir-ml-aljabr` module, but it is a general tensor/NN bridge and is not the SafeTensors quantizer integration.

The settings file also contains compatibility includes for:
- `:extensions:aljabr:tafkir-aljabr-integration`
- `:extensions:aljabr:tafkir-trainer-aljabr`

Those project directories are not currently present in the repository tree. The integration project should be created at the first path rather than adding more aliases to the general `tafkir-ml-aljabr` module.

Tafkir's quantizer implementations currently live under `inference/quantization`:
- `tafkir-quantizer-gptq`
- `tafkir-quantizer-awq`
- `tafkir-quantizer-autoround`
- `tafkir-quantizer-turboquant`

TurboQuant also contains the `BnBDequantizer` currently used by Aljabr's BitsAndBytes adapter.

## Ownership and migration inventory

The following adapters currently live in Aljabr's `core/aljabr-safetensor-quantization` module but import Tafkir implementation classes. They belong in the Tafkir integration extension:

| Current adapter | Tafkir implementation dependency | Target owner |
| --- | --- | --- |
| `GPTQQuantizerAdapter` | `tafkir-quantizer-gptq` | Tafkir |
| `AWQQuantizerAdapter` | `tafkir-quantizer-awq` | Tafkir |
| `AutoRoundQuantizerAdapter` | `tafkir-quantizer-autoround` | Tafkir |
| `TurboQuantAdapter` | `tafkir-quantizer-turboquant` | Tafkir |
| `BnBQuantizerAdapter` | `BnBDequantizer` from `tafkir-quantizer-turboquant` | Tafkir |

Do not remove or replace these adapters until equivalent Tafkir-owned implementations are available and discoverable by Aljabr. Preserve the existing strategy names and tensor metadata behavior during migration.

Aljabr's native `GPTQQuantizer`, `FP8Quantizer`, and `INT8Quantizer` are Aljabr-owned implementations and are not part of this move.

## Target module

Create `extensions/aljabr/tafkir-aljabr-integration` as a separately publishable Tafkir extension.

It should:
1. Depend on Aljabr's published SafeTensors quantization API/SPI.
2. Depend on the Tafkir quantizer modules it adapts.
3. Implement Aljabr's `tech.kayys.aljabr.safetensor.quantization.quantizer.Quantizer` contract.
4. Register implementations using Java `ServiceLoader` metadata (or an equally explicit extension registration mechanism supported by Aljabr).
5. Keep Tafkir implementation classes out of Aljabr's source tree and dependency graph.

The integration must not make the base Tafkir modules depend on Aljabr's application/runtime implementation. The extension is the optional bridge.

## Aljabr-side requirements

Aljabr's SPI and registry must support optional implementations without importing or constructing Tafkir classes. In particular:

- The `Quantizer` contract remains owned by Aljabr.
- Registry discovery must load optional providers without requiring them to be installed.
- Built-in Aljabr quantizers remain available when the Tafkir extension is absent.
- Duplicate names and provider loading failures must be handled deterministically and diagnostically.
- `QuantizationEngine` must not directly instantiate Tafkir adapters.
- The default Aljabr artifact must not declare Tafkir quantizer dependencies or include a Tafkir composite build.

## Migration sequence

1. Add this Tafkir-owned extension module and its focused tests.
2. Move the five Tafkir-backed adapters into the extension, retaining their public strategy/name behavior.
3. Add service registration and Aljabr registry discovery.
4. Remove the corresponding adapter classes and direct Tafkir construction from Aljabr.
5. Remove the four Tafkir quantizer dependencies from Aljabr's SafeTensors quantization module.
6. Remove the optional Tafkir composite-build wiring from Aljabr after confirming no remaining Aljabr project needs it.
7. Add dependency-boundary checks in both repositories.

## Acceptance criteria

- Aljabr builds and runs with no Tafkir artifacts on its classpath.
- Installing the Tafkir integration extension makes the moved quantizers discoverable.
- Aljabr native quantizers continue to work without the extension.
- Each moved adapter has tests for discovery, supported strategies, quantize/dequantize behavior, and unsupported configuration handling.
- Dependency reports and source scans show no Tafkir references in Aljabr production modules.
- Both standalone artifact consumption and the sibling composite-build development workflow are validated.

## Known limitations in the current implementation

This migration is an ownership and dependency-boundary correction; it is not a claim that all current adapter algorithms are production-complete. Some existing adapter code contains placeholder or simplified quantization paths. Moving code must not be treated as algorithmic validation. Correctness, tensor packing/layout compatibility, metadata lifetime, and large-tensor memory behavior need explicit tests before the adapters are promoted as production-ready.
