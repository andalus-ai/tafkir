# Tafkir

[![CI](https://github.com/bhangun/tafkir/actions/workflows/ci.yml/badge.svg)](https://github.com/bhangun/tafkir/actions/workflows/ci.yml)
[![Release](https://github.com/bhangun/tafkir/actions/workflows/release.yml/badge.svg)](https://github.com/bhangun/tafkir/actions/workflows/release.yml)
[![GitHub Packages](https://img.shields.io/badge/Packages-GitHub%20Packages-181717?logo=github)](https://github.com/bhangun/tafkir/packages)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Java 25](https://img.shields.io/badge/Java-25+-orange.svg)](https://adoptium.net/)
[![JDK Vector API](https://img.shields.io/badge/JDK-Vector%20API-5E5E5E)](https://openjdk.org/jeps/438)

A Java machine learning framework built on the [Alkhawarizm](https://github.com/bhangun/alkhawarizm) compute engine.

## Architecture

```
┌─────────────────────────────────────────┐
│              Tafkir                     │
│  (training APIs, CLI, examples)       │
├─────────────────────────────────────────┤
│           Alkhawarizm Engine                 │
│  (tensor ops, autograd, backends)     │
│  • CPU: Vector API SIMD                 │
│  • CUDA: GPU kernels                    │
│  • Metal: Apple Silicon                 │
│  • Blackwell: NVIDIA B100/B200          │
└─────────────────────────────────────────┘
```

## Prerequisites

- JDK 25+
- Alkhawarizm checked out at `../alkhawarizm` (composite build)

## Quick Start

```bash
git clone https://github.com/bhangun/alkhawarizm.git ../alkhawarizm
git clone https://github.com/bhangun/tafkir.git
cd tafkir
./gradlew :trainer:tafkir-trainer-alkhawarizm:test
```

## Status

| Feature | Status |
|---------|--------|
| Tensor ops (CPU, SIMD) | Working |
| Autograd (graph-based) | Working |
| MLP training | Working |
| CNN (conv2d, pool) | Working |
| Data loaders (MNIST) | Working |
| LLM inference (GGUF) | Working (via llama.cpp) |
| CUDA training | In progress |
| Distributed training | Planned |

## Example: XOR

```java
TafkirSequential model = new TafkirSequential(
    new TafkirLinear(2, 2),
    new TafkirReLU(),
    new TafkirLinear(2, 1)
);

TafkirTrainer trainer = new TafkirTrainer(
    model,
    new TafkirMSELoss(),
    new TafkirAdam(model.parameters(), 0.1f),
    1000
);

trainer.fit(x, y);
```

## Example: MNIST CNN

```java
TafkirSequential model = new TafkirSequential(
    new TafkirConv2d(1, 32, 3, 1, 1, 1, 1),
    new TafkirReLU(),
    new TafkirMaxPool2d(2, 2, 0),
    new TafkirConv2d(32, 64, 3, 1, 1, 1, 1),
    new TafkirReLU(),
    new TafkirMaxPool2d(2, 2, 0),
    new TafkirFlatten(),
    new TafkirLinear(3136, 128),
    new TafkirReLU(),
    new TafkirLinear(128, 10)
);
```

## Modules

| Module | Description |
|--------|-------------|
| `ml:tafkir-ml-alkhawarizm` | Tensor API wrapping Alkhawarizm backends |
| `trainer:tafkir-trainer-alkhawarizm` | Training loop, optimizers, losses |
| `data:tafkir-data` | Dataset loaders (MNIST, etc.) |
| `distributed:tafkir-distributed` | Multi-threaded data parallel training |
| `checkpoint:tafkir-checkpoint` | Model save/load |
| `tafkir-cli` | Command-line inference tool |
