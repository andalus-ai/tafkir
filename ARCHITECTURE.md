# Tafkir Architecture & Ecosystem Boundaries

## 1. Executive Summary & Vision

**Tafkir** is the dedicated **Machine Learning Training, Neural Modeling, and Fine-Tuning framework** in pure Java. It forms the training pillar of the Wayang ML stack, working symbiotically with **Alkhawarizm** (foundation tensor engine & hardware backends) and **Gollek** (production inference runtime).

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Wayang Platform                               │
│                   Agentic AI & Workflow Orchestration                  │
├───────────────────────────────────┬────────────────────────────────────┤
│              Tafkir               │               Gollek               │
│   (Training & Model Development)  │    (Inference & Serving Runtime)   │
│  • Dataset & DataLoader Pipelines │  • High-throughput Token Generation│
│  • Neural Network Modules & Blocks│  • GGUF / ONNX / LiteRT Runners    │
│  • Autograd Graph Tape Execution  │  • Paged KV Cache & Batching       │
│  • Optimizers (Adam, SGD, RMSprop)│  • Continuous Batch Scheduler      │
│  • Loss Functions & Schedulers    │  • Model Database & Panache Repo   │
│  • Quantization-Aware Training    │  • Local SDK & REST/gRPC Serving   │
├───────────────────────────────────┴────────────────────────────────────┤
│                       Alkhawarizm Foundation                           │
│  • Canonical Tensor Primitives (Tensor, DefaultTensor, Shape, DType)   │
│  • Accelerated Hardware Backends (CPU SIMD, Apple Metal, NVIDIA CUDA)  │
│  • Shared Inference, Model, and Tokenizer SPIs                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Platform Comparison & Separation of Concerns

| Concern | Alkhawarizm | Tafkir | Gollek |
| :--- | :--- | :--- | :--- |
| **Primary Role** | Compute engine & Hardware backends | Training & Model Development | Production Inference & Serving |
| **Python Ecosystem Analog** | PyTorch C++/LibTorch / CUDA / Metal | PyTorch Lightning / TorchVision / Keras | vLLM / TensorRT-LLM / llama.cpp server |
| **Maven Group ID** | `tech.kayys.alkhawarizm` | `tech.kayys.tafkir` | `tech.kayys.gollek` |
| **Java Package** | `tech.kayys.alkhawarizm.*` | `tech.kayys.tafkir.*` | `tech.kayys.gollek.*` |
| **Hardware Execution** | ✅ Implements native/SIMD/GPU kernels | ❌ Delegates to Alkhawarizm backends | ❌ Delegates to Alkhawarizm & runners |
| **Autograd & Gradients** | Base autograd ops | Full dynamic graph tape & backward pass | ❌ Forward-pass only (NoGrad) |
| **Quantization** | Low-level quantization ops | Quantization-Aware Training (QAT) | Post-Training Quantization (PTQ) serving |
| **Model Registry** | Basic metadata SPI | Dataset loaders, training checkpoints | Distributed model storage, cache, Panache |

---

## 3. End-to-End Workflow: Train → Export → Serve

The canonical dataflow across the three platforms is:

```
[Datasets / DataLoaders]
         │
         ▼
[Tafkir Training Loop] ──(backprop)──► [Alkhawarizm Compute Backends]
         │
         ▼
   (Trained Weights)
         │
         ▼
[Model Exporter] ────────► [Export to SafeTensors / GGUF / ONNX]
                                       │
                                       ▼
                       [Gollek Production Serving Engine]
                                       │
                                       ▼
                         [High-Throughput User APIs]
```

```java
// 1. Train model in Tafkir
TafkirTrainer trainer = new TafkirTrainer(
    myModel,
    new TafkirCrossEntropyLoss(),
    new TafkirAdam(myModel.parameters(), 0.001f),
    100
);
trainer.fit(trainLoader, valLoader);

// 2. Export weights
ModelExporter.export(trainer.getModel(), Path.of("checkpoints/model.safetensors"));

// 3. Serve in production with Gollek
// Gollek loads model.safetensors via SafeTensorsRunner and serves at line-rate.
```

---

## 4. Module Taxonomy

### `data/` — Data Ingestion & Transformation
* `data/core/tafkir-data`: Core dataset abstractions, batch samplers, and streaming iterators.
* `data/core/tafkir-learn-data`: Out-of-the-box datasets (MNIST, CIFAR, NLP benchmarks).

### `models/` & `ml/` — Neural Layers & Architectures
* `models/nn-core/tafkir-nn`: Foundational neural building blocks (`Sequential`, `Linear`, `Conv2d`, `Dropout`, `LayerNorm`, `Attention`).
* `models/nn-vision/tafkir-nn-cnn`: Convolutional vision blocks and backbones.
* `models/nn-rnn/tafkir-nn-rnn`: Recurrent structures (LSTM, GRU, RNN).
* `models/nn-multimodal/tafkir-nn-multimodal`: Multi-modal projection and cross-attention blocks.
* `ml/core/tafkir-ml-core`: Scikit-learn style estimators, metrics, and cross-validation utilities.

### `training/` — The Training Engine
* `training/api/tafkir-trainer-api`: Contracts for `Trainer`, `TrainingListener`, `EpochSession`, and callbacks.
* `training/loop/tafkir-trainer`: Main training loop engine supporting gradient accumulation and validation steps.
* `training/optim/tafkir-learn-optim`: First-class optimizers (`Adam`, `AdamW`, `SGD`, `RMSprop`, learning rate schedulers).

### `runtime/` — Autograd & Graph Engine
* `runtime/autograd/tafkir-ml-autograd`: Reverse-mode automatic differentiation tape executed over Alkhawarizm tensors.
* `runtime/core/tafkir-distributed`: Data-parallel training primitives (DDP wrapper, gradient synchronization).

### `inference/quantization/` — Compression & Quantization
* `tafkir-quantizer-autoround`, `tafkir-quantizer-awq`, `tafkir-quantizer-gptq`, `tafkir-quantizer-turboquant`: Quantization algorithms and QAT tooling.

---

## 5. Architectural Boundary Enforcement

Tafkir enforces 4 core architectural boundaries checked by `./scripts/check-boundaries.sh`:

1. **No Phantom Stub Directories**: Tafkir does not contain fake `backend/` or `stubs/` directories. All hardware compute is delegated to Alkhawarizm.
2. **Strict Group ID Isolation**: All Tafkir modules are published exclusively under `group = "tech.kayys.tafkir"`. No module declares foreign groups (`tech.kayys.alkhawarizm` or `tech.kayys.gollek`).
3. **Canonical Dependency Coordinates**: Tafkir modules depend on Alkhawarizm via published snapshot coordinates (`tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT`, etc.), resolved automatically in local development via Gradle composite build substitution.
4. **Clean Decoupling from Gollek**: Tafkir does not bind directly to Gollek runtime internals, allowing both frameworks to evolve and release independently.
