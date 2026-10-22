# Tafkir

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Java 25](https://img.shields.io/badge/Java-25+-orange.svg)](https://adoptium.net/)
[![JDK Vector API](https://img.shields.io/badge/JDK-Vector%20API-5E5E5E)](https://openjdk.org/jeps/438)

A comprehensive machine learning, neural modeling, and deep learning training framework in pure Java, built on top of the **[Aljabr](https://github.com/bhangun/aljabr)** foundation engine.

---

## 🏛️ Ecosystem Architecture

Tafkir is the **Training & Modeling framework** of the Wayang ML stack:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Wayang Platform                               │ ← Agent orchestration, MCP tools
├───────────────────────────────────┬────────────────────────────────────┤
│              Tafkir               │               Aqli               │
│   (Training & Model Development)  │    (Inference & Serving Runtime)   │
│  • Datasets & DataLoaders         │  • GGUF / ONNX / LiteRT Runners    │
│  • Neural Network Modules         │  • KV Cache & Continuous Batching  │
│  • Optimizers & Loss Functions    │  • Model Distribution (HF/Panache) │
│  • Autograd Graph & Tape Engine   │  • Local SDK & Serving APIs        │
│  • Quantization-Aware Training    │  • Post-Training Quantization      │
├───────────────────────────────────┴────────────────────────────────────┤
│                       Aljabr Foundation                           │
│  • Tensor primitives (DefaultTensor, Shape, DType, Memory Layout)      │
│  • Hardware Backends (CPU Vector SIMD, Apple Metal, NVIDIA CUDA)      │
│  • Core Serving & Inference SPIs                                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Capabilities

* **PyTorch-like Neural Modules**: `TafkirSequential`, `TafkirLinear`, `TafkirConv2d`, `TafkirReLU`, `TafkirMaxPool2d`, Attention, and Custom Layers.
* **Autograd Tape & Computational Graph**: Dynamic graph construction with automated backward differentiation (`tafkir-ml-autograd`).
* **Complete Training Engine**: Production training loops, validation loops, learning rate schedulers, and optimizers (`Adam`, `AdamW`, `SGD`, `RMSprop`).
* **Loss Functions**: MSE, Cross-Entropy, Focal Loss, Triplet Loss, and custom losses.
* **Datasets & Transforms**: Out-of-the-box data loaders and preprocessors (MNIST, text tokenization, audio features).
* **Quantization & Compilers**: Quantization-aware training (QAT), TurboQuant, AWQ, and neural graph compilation.

---

## 💻 Example: Training an MLP

```java
// Define architecture
TafkirSequential model = new TafkirSequential(
    new TafkirLinear(784, 128),
    new TafkirReLU(),
    new TafkirLinear(128, 10)
);

// Configure trainer
TafkirTrainer trainer = new TafkirTrainer(
    model,
    new TafkirCrossEntropyLoss(),
    new TafkirAdam(model.parameters(), 0.001f),
    10 // epochs
);

// Train
trainer.fit(trainDataLoader, validationDataLoader);
```

---

## 📦 Project Structure

| Category | Modules | Description |
| :--- | :--- | :--- |
| **Data Pipelines** | `data/core/tafkir-data`<br>`data/core/tafkir-learn-data` | Datasets, data loaders, batching, and data transforms. |
| **Neural Layers** | `models/nn-core/tafkir-nn`<br>`models/nn-vision/tafkir-nn-cnn`<br>`models/nn-rnn/tafkir-nn-rnn`<br>`models/nn-multimodal/*` | Dense, convolutional, recurrent, and multimodal layers. |
| **Training Engine** | `training/api/tafkir-trainer-api`<br>`training/loop/tafkir-trainer`<br>`training/optim/tafkir-learn-optim` | Training loops, gradient accumulation, and optimizers. |
| **Autograd** | `runtime/autograd/tafkir-ml-autograd` | Dynamic graph execution and automatic backward differentiation. |
| **Quantization** | `inference/quantization/tafkir-quantizer-*` | AWQ, GPTQ, AutoRound, and TurboQuant algorithms. |
| **Tooling & CLI** | `ecosystem/cli/tafkir-cli`<br>`ecosystem/jupyter/tafkir-jupyter-kernel` | Training CLI and interactive Jupyter kernel integration. |

---

## 🛡️ Architectural Boundaries

To ensure clean architecture across the ecosystem:
1. **Foundation Dependency**: Tafkir consumes Aljabr foundation artifacts (`tech.kayys.aljabr:aljabr-tensor`, etc.) without maintaining duplicate backend implementations.
2. **Serving Separation**: Real-time production inference serving is owned by Aqli; Tafkir focuses on training, evaluation, and model synthesis.
3. **Automated Verification**: Run `./scripts/check-boundaries.sh` to enforce boundary rules in CI.
