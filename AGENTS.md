# AGENTS.md — Tafkir contributor guidance

## Product vision and ownership

Tafkir is the training, fine-tuning, adaptation, and evaluation layer of the Java-first AQLI + Aljabr + Tafkir AI/ML ecosystem. The shared product-level direction is documented in the [Triad Ecosystem Vision](https://github.com/andalus-ai/aqli/blob/main/docs/architecture/TRIAD_ECOSYSTEM_VISION.md).

The ecosystem aims to provide practical Java capability coverage for workflows commonly assembled from PyTorch, Hugging Face Transformers, scikit-learn, training toolkits, and inference servers. This is a strategic target, not a claim of complete or drop-in parity. Track actual support and maturity explicitly.

Tafkir owns:
- Training loops, trainer orchestration, training lifecycle, and experiment/run workflows.
- Fine-tuning, adaptation, quantization-aware training, and other training-time workflows where implemented.
- Dataset, data-loader, batching, sampling, and training-time preprocessing pipelines.
- Evaluation and training-oriented metrics/workflows.
- Training CLI, scripting, and notebook/Jupyter integration where provided.
- Export and handoff of portable model artifacts for downstream use.

Keep these boundaries explicit:
- **Aljabr** owns reusable computation: tensors, numerical semantics, operators/kernels, model foundations, autograd/optimizer primitives, backend/provider SPIs, formats, and reusable ML algorithms. Tafkir should consume Aljabr APIs rather than maintain duplicate tensor/backend implementations.
- **AQLI** owns inference serving: request execution, model serving lifecycle, scheduling, streaming, deployment, SDK/API serving, and production operations. Tafkir must not become an AQLI training mode or own production inference-serving policy.
- Tafkir must not require AQLI internals to train, fine-tune, or evaluate models.
- Aljabr must not depend on Tafkir. Tafkir may depend on published Aljabr APIs; avoid circular dependencies and sibling-repository implementation coupling.

## Consumption modes and developer experience

The same Tafkir training APIs and artifacts should be usable through:
- Embedded Java libraries in applications and research programs.
- Java SDK-style APIs and ordinary Java entry points.
- CLI workflows for training, evaluation, and artifact operations.
- JBang and repeatable Java scripts for experiments, automation, and CI.
- JShell/Jupyter/JVM notebook workflows where supported.
- Automated training jobs and production pipelines that hand off artifacts to AQLI.

These are entry points over shared APIs, not separate training implementations. Do not make a notebook, CLI, or script the exclusive owner of a capability. Keep examples runnable against published artifacts where practical.

## Architecture and implementation rules

- Reuse Aljabr's stable public tensor, model, autograd, optimizer, operator, and backend contracts wherever they satisfy Tafkir's needs.
- Keep training-specific orchestration and lifecycle in Tafkir; do not push it into Aljabr merely to share implementation.
- Keep request-serving, online scheduling, inference sessions, streaming transport, and production deployment policy in AQLI.
- Use typed, versioned contracts for datasets, training configuration, evaluation, checkpoints, and exported artifacts. Avoid permanent untyped parameter maps.
- Keep optional integrations optional; a basic training or evaluation workflow must not pull in AQLI serving infrastructure.
- Preserve portable artifact formats and document compatibility requirements for the Aljabr model APIs and AQLI runtime.
- Maintain explicit capability status: implemented, partial, experimental, planned, or unsupported. An API or module declaration alone does not prove working training support.
- Back capability claims with correctness tests, runnable examples, documented data/model/backend constraints, and reproducible benchmarks where relevant.
- Inspect actual Gradle modules and current implementations before creating or renaming module boundaries. Proposed architecture is a target, not evidence that a module exists.
- Keep public APIs independently consumable and document JDK, dependency, device, and native-runtime requirements.

## Cross-project acceptance criteria

- Tafkir can train/adapt/evaluate using Aljabr foundations without depending on AQLI internals.
- Training outputs are portable and can be consumed through Aljabr model APIs and deployed through AQLI where the format/model/backend combination is supported.
- The same training semantics are available through relevant Java, CLI, script, and notebook entry points.
- Aljabr remains independent of Tafkir; AQLI's ordinary inference path remains independent of Tafkir.
- Documentation distinguishes demonstrated capability from roadmap intent and records limitations honestly.

## Shared memory and zero-copy consumption

- Aljabr owns the memory-store and borrow/lease contract. Tafkir must not invent a parallel allocator or depend on Aljabr provider implementation internals beyond explicitly optional provider modules.
- When a provider exposes memory through `MemoryStoreLease`, keep the lease open for the entire period in which Java code, FFM downcalls, or native code can access the borrowed segment. Close it deterministically with try-with-resources after the final access.
- Never retain a `MemorySegment` obtained from a lease after that lease closes. Do not pass the segment to asynchronous work unless the lease lifetime is transferred and managed for that work.
- Do not use the legacy unleased `getZeroCopy(...)` API for new code. Prefer a provider capability such as `BorrowableMemoryStore`; if a provider cannot guarantee the borrow lifetime, copy into caller-owned memory and describe that copy honestly.
- A memory-store borrow is not a checkpoint, durable artifact, dataset store, or training-state persistence mechanism. Tafkir owns checkpoint lifecycle and should use portable filesystem/object-store artifact formats for durable training outputs.
- Keep native LibTorch execution tests separate from portable memory-contract tests. Do not catch broad runtime exceptions and report the test as successful; explicitly skip native tests when their required native runtime is not provisioned.
