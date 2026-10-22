(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: README.md
================================================
# Tafkir

[![CI](https://github.com/bhangun/tafkir/actions/workflows/ci.yml/badge.svg)](https://github.com/bhangun/tafkir/actions/workflows/ci.yml)
[![Release](https://github.com/bhangun/tafkir/actions/workflows/release.yml/badge.svg)](https://github.com/bhangun/tafkir/actions/workflows/release.yml)
[![GitHub Packages](https://img.shields.io/badge/Packages-GitHub%20Packages-181717?logo=github)](https://github.com/bhangun/tafkir/packages)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Java 25](https://img.shields.io/badge/Java-25+-orange.svg)](https://adoptium.net/)
[![JDK Vector API](https://img.shields.io/badge/JDK-Vector%20API-5E5E5E)](https://openjdk.org/jeps/438)

A Java machine learning framework built on the [Aljabr](https://github.com/bhangun/aljabr) compute engine.

## Architecture

```
┌─────────────────────────────────────────┐
│              Tafkir                     │
│  (training APIs, CLI, examples)       │
├─────────────────────────────────────────┤
│           Aljabr Engine                 │
│  (tensor ops, autograd, backends)     │
│  • CPU: Vector API SIMD                 │
│  • CUDA: GPU kernels                    │
│  • Metal: Apple Silicon                 │
│  • Blackwell: NVIDIA B100/B200          │
└─────────────────────────────────────────┘
```

## Prerequisites

- JDK 25+
- Aljabr checked out at `../aljabr` (composite build)

## Quick Start

```bash
git clone https://github.com/bhangun/aljabr.git ../aljabr
git clone https://github.com/bhangun/tafkir.git
cd tafkir
./gradlew :trainer:tafkir-trainer-aljabr:test
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
| `ml:tafkir-ml-aljabr` | Tensor API wrapping Aljabr backends |
| `trainer:tafkir-trainer-aljabr` | Training loop, optimizers, losses |
| `data:tafkir-data` | Dataset loaders (MNIST, etc.) |
| `distributed:tafkir-distributed` | Multi-threaded data parallel training |
| `checkpoint:tafkir-checkpoint` | Model save/load |
| `tafkir-cli` | Command-line inference tool |



================================================
FILE: ARCHITECTURE.md
================================================
# Tafkir & Aqli Architecture Overview

## Project Evolution

This document explains the relationship between **Tafkir** and **Aqli**, two related but distinct platforms in our ML/AI ecosystem.

### Historical Context

The project originated as **Aqli**, a high-performance inference and serving engine for large language models. As the platform matured, the training capabilities grew into a comprehensive machine learning framework, which was spun off as **Tafkir**.

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML/AI Ecosystem                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │     Aqli       │         │     Tafkir       │        │
│  │                  │         │                  │        │
│  │  Inference &     │         │  Training        │        │
│  │  Serving Engine  │         │  Framework       │        │
│  │                  │         │                  │        │
│  │  - GGUF Runtime  │         │  - Neural Nets   │        │
│  │  - SafeTensors   │         │  - Autograd      │        │
│  │  - Model Serving │         │  - Training Loop │        │
│  │  - CLI Tools     │◄───────►│  - Optimizers    │        │
│  │  - Quantization  │         │  - Data Loading  │        │
│  └──────────────────┘         │  - Strategies    │        │
│                               └──────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Platform Comparison

| Feature | Aqli | Tafkir |
|---------|--------|--------|
| **Primary Focus** | Inference & Serving | Training & Development |
| **Use Case** | Deploy models to production | Build and train custom models |
| **Package Namespace** | `tech.kayys.aqli.*` | `tech.kayys.tafkir.*` |
| **CLI** | ✅ Aqli CLI | ✅ Tafkir CLI (shared heritage) |
| **Model Support** | Pre-trained models | Custom + pre-trained |
| **Quantization** | ✅ PTQ (GPTQ, AWQ, etc.) | ✅ QAT + PTQ |
| **Training APIs** | ❌ | ✅ PyTorch-like |
| **Jupyter Support** | Limited | ✅ Full integration |
| **JBang Scripts** | Inference examples | Training + inference |

### Shared Components

Both platforms share several components due to their common heritage:

1. **CLI Tool**: The command-line interface evolved from Aqli and is now integrated into Tafkir for end-to-end workflows
2. **Model Registry**: Shared model discovery and management
3. **Quantization Modules**: GPTQ, AWQ, QUIP, TurboQuant implementations
4. **Format Support**: GGUF, SafeTensors, TorchScript, ONNX

### Package Structure

#### Aqli Packages (`tech.kayys.aqli.*`)
- Model family plugins (e.g., `tafkir-model-llama`, `tafkir-model-bert`)
- Inference runtime providers
- Serving infrastructure
- Legacy SPI implementations

#### Tafkir Packages (`tech.kayys.tafkir.*`)
- `tafkir.ml.*` - Core ML abstractions (tensors, autograd, nn modules)
- `tafkir.trainer.*` - Training framework APIs
- `tafkir.training.*` - Training implementations
- `tafkir.sdk.*` - High-level SDK for developers
- `tafkir.cli.*` - Command-line interface

### When to Use Which

**Use Aqli when:**
- You need to deploy pre-trained models to production
- You want maximum inference performance
- You're building serving infrastructure
- You need quantization for existing models

**Use Tafkir when:**
- You want to train custom models from scratch
- You need PyTorch-like training workflows in Java
- You're experimenting with new architectures
- You want Jupyter notebook support
- You need end-to-end ML pipelines (train → evaluate → deploy)

### Integration Example

A typical workflow might use both platforms:

```java
// Step 1: Train custom model with Tafkir
Trainer trainer = Trainers.canonicalBuilder()
    .model(myCustomArchitecture)
    .dataset(trainingData)
    .build();
trainer.fit();

// Step 2: Export to ONNX or SafeTensors
ModelExporter.export(trainer.getModel(), "model.safetensors");

// Step 3: Deploy with Aqli for high-performance serving
// (or use Tafkir CLI which includes Aqli inference capabilities)
```

### Migration Path

For existing Aqli users:
- Model plugins under `tech.kayys.aqli.*` continue to work
- New training features should use `tech.kayys.tafkir.*` packages
- CLI commands are compatible across both platforms
- Gradle module names use `tafkir-*` prefix for consistency

### Version Compatibility

| Component | Version Scheme | Release Status |
|-----------|---------------|----------------|
| Tafkir Core | 0.3.0-SNAPSHOT | Active development |
| Aqli Inference | 1.x.x | Stable |
| Shared CLI | Unified | Latest in Tafkir |
| Model Plugins | Varies by model | Mixed |

### Future Direction

- **Tafkir** will continue expanding training capabilities, adding more PyTorch/scikit-learn parity
- **Aqli** will focus on inference optimization and serving performance
- Both platforms will maintain compatibility through shared formats (ONNX, SafeTensors, GGUF)
- The CLI will remain unified for seamless train-to-deploy workflows

---

**Questions?** Open an issue on GitHub or check the main [README.md](../README.md) for getting started guides.



================================================
FILE: build-macos.sh
================================================
#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/build-macos.sh"

if [[ ! -x "$SCRIPT_PATH" ]]; then
  echo "Missing $SCRIPT_PATH"
  echo "Run ./create-builder-script.sh to generate it."
  exit 1
fi

exec "$SCRIPT_PATH" "$@"



================================================
FILE: build.gradle.kts
================================================
import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.publish.PublishingExtension
import org.gradle.api.publish.maven.MavenPublication
import org.gradle.api.tasks.javadoc.Javadoc
import org.gradle.external.javadoc.StandardJavadocDocletOptions

buildscript {
    repositories {
        google()
        mavenCentral()
        mavenLocal()
    }
    dependencies {
        classpath("io.smallrye:jandex:3.5.3")
    }
}

plugins {
    java
    `maven-publish`
    id("io.quarkus") version "3.32.2" apply false
    id("io.github.gradle-nexus.publish-plugin") version "2.0.0" apply false
}

extra["tafkirVersion"] = "0.1.1"
extra["quarkusVersion"] = "3.32.2"

allprojects {
    group = "tech.kayys.tafkir"
    version = rootProject.extra["tafkirVersion"] as String

    repositories {
        google()
        mavenCentral()
        mavenLocal()
    }
}

subprojects {
    apply(plugin = "java")
    apply(plugin = "maven-publish")

    val quarkusVersion = rootProject.extra["quarkusVersion"] as String
    val mutinyVersion = "2.5.5"
    val smallryeMutinyVertxVersion = "3.15.1"
    val caffeineVersion = "3.1.8"
    val commonsCollectionsVersion = "4.4"
    val jakartaValidationVersion = "3.0.2"
    val jakartaEnterpriseVersion = "4.0.1"
    val jakartaInjectVersion = "2.0.1"
    val reactiveStreamsVersion = "1.0.4"
    val jbossLoggingVersion = "3.6.1.Final"
    val jacksonVersion = "2.16.1"
    val junitJupiterVersion = "5.10.2"
    val junitPlatformVersion = "1.10.2"
    val mockitoJupiterVersion = "5.14.2"

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(25))
        }
    }

    dependencies {
        add("testRuntimeOnly", "org.junit.platform:junit-platform-launcher:$junitPlatformVersion")
        add("testImplementation", platform("org.junit:junit-bom:$junitJupiterVersion"))
        add("testImplementation", "org.junit.jupiter:junit-jupiter")
        add("testImplementation", "org.assertj:assertj-core:3.25.3")
    }

    tasks.withType<JavaCompile> {
        options.encoding = "UTF-8"
        options.compilerArgs.add("--enable-preview")
        options.compilerArgs.add("--add-modules=jdk.incubator.vector")
    }

    tasks.withType<Javadoc>().configureEach {
        options {
            this as StandardJavadocDocletOptions
            addBooleanOption("Xdoclint:none", true)
            addStringOption("-add-modules", "jdk.incubator.vector")
        }
    }

    tasks.withType<Test>().configureEach {
        jvmArgs(
            "--enable-preview",
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
        )
        useJUnitPlatform()
    }

    tasks.withType<JavaExec>().configureEach {
        jvmArgs(
            "--enable-preview",
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
        )
    }

    val sourceSets = extensions.getByType(SourceSetContainer::class.java)
    val mainSourceSet = sourceSets.named("main")
    val jandexOutputDir = layout.buildDirectory.dir("generated/jandex/main")

    val generateJandexIndex = tasks.register("generateJandexIndex") {
        dependsOn(tasks.named("compileJava"), tasks.named("processResources"))

        val classesDirs = mainSourceSet.map { it.output.classesDirs }
        val indexFile = jandexOutputDir.map { it.file("META-INF/jandex.idx") }

        inputs.files(classesDirs)
        outputs.file(indexFile)

        doLast {
            val indexer = org.jboss.jandex.Indexer()

            classesDirs.get().files
                .filter { it.exists() }
                .forEach { classesDir ->
                    classesDir.resolve("META-INF/jandex.idx").let { staleIndex ->
                        if (staleIndex.exists()) {
                            staleIndex.delete()
                        }
                    }
                    classesDir.walkTopDown()
                        .filter { it.isFile && it.extension == "class" }
                        .forEach { classFile ->
                            classFile.inputStream().use(indexer::index)
                        }
                }

            val indexOutput = indexFile.get().asFile
            indexOutput.parentFile.mkdirs()
            indexOutput.outputStream().use { output ->
                org.jboss.jandex.IndexWriter(output).write(indexer.complete())
            }
        }
    }

    mainSourceSet.configure {
        output.dir(mapOf("builtBy" to generateJandexIndex), jandexOutputDir)
    }

    tasks.named("classes") {
        dependsOn(generateJandexIndex)
    }

    configurations.configureEach {
        resolutionStrategy.eachDependency {
            if (requested.group == "io.quarkus" && requested.version.isNullOrBlank()) {
                useVersion(quarkusVersion)
            }
            if (requested.group == "io.smallrye.reactive"
                && requested.name == "mutiny"
                && requested.version.isNullOrBlank()) {
                useVersion(mutinyVersion)
            }
            if (requested.group == "com.fasterxml.jackson.core"
                && (requested.name == "jackson-core" || requested.name == "jackson-databind")
                && requested.version.isNullOrBlank()) {
                useVersion(jacksonVersion)
            }
        }
    }

    // Apply nexus publishing plugin to root project only
    if (project == rootProject) {
        apply(plugin = "io.github.gradle-nexus.publish-plugin")
        extensions.configure<io.github.gradlenexus.publishplugin.NexusPublishExtension> {
            repositories {
                sonatype {
                    nexusUrl.set(uri("https://s01.oss.sonatype.org/service/local/"))
                    snapshotRepositoryUrl.set(uri("https://s01.oss.sonatype.org/content/repositories/snapshots/"))
                }
            }
        }
    }

    afterEvaluate {
        extensions.configure<PublishingExtension>("publishing") {
            publications {
                if (publications.findByName("mavenJava") == null) {
                    create<MavenPublication>("mavenJava") {
                        from(components["java"])
                        
                        // Add POM metadata for Maven Central
                        pom {
                            name.set(project.name)
                            description.set("Tafkir ML/AI Training Framework - ${project.name}")
                            url.set("https://github.com/bhangun/tafkir")
                            
                            licenses {
                                license {
                                    name.set("Apache License 2.0")
                                    url.set("https://www.apache.org/licenses/LICENSE-2.0")
                                }
                            }
                            
                            developers {
                                developer {
                                    id.set("bhangun")
                                    name.set("Tafkir Contributors")
                                    email.set("dev@tafkir.tech")
                                }
                            }
                            
                            scm {
                                connection.set("scm:git:git://github.com/bhangun/tafkir.git")
                                developerConnection.set("scm:git:ssh://github.com/bhangun/tafkir.git")
                                url.set("https://github.com/bhangun/tafkir")
                            }
                        }
                    }
                }
            }
            
            repositories {
                // Maven Central will be configured via nexus publish plugin
                // This is kept for local development
                mavenLocal()

                // GitHub Packages Maven registry requires: https://maven.pkg.github.com/OWNER/REPOSITORY
                val repoUrlProperty = providers.gradleProperty("deployment.repo.url").orNull
                val githubRepo = System.getenv("GITHUB_REPOSITORY")
                    ?: providers.gradleProperty("github.repository").orNull
                    ?: "bhangun/tafkir"
                val githubActor = System.getenv("GITHUB_ACTOR") ?: providers.gradleProperty("gpr.user").orNull
                val githubToken = System.getenv("GITHUB_TOKEN") ?: providers.gradleProperty("gpr.key").orNull
                if (!githubActor.isNullOrBlank() && !githubToken.isNullOrBlank()) {
                    maven {
                        name = "GitHubPackages"
                        url = uri(repoUrlProperty ?: "https://maven.pkg.github.com/$githubRepo")
                        credentials {
                            username = githubActor
                            password = githubToken
                        }
                    }
                }
            }
        }
        
        // Add Javadoc and Sources jars
        val javadocJar = tasks.register<Jar>("javadocJar") {
            dependsOn(tasks.named("javadoc"))
            archiveClassifier.set("javadoc")
            from(tasks.named("javadoc").map { (it as Javadoc).destinationDir })
        }
        
        val sourcesJar = tasks.register<Jar>("sourcesJar") {
            archiveClassifier.set("sources")
            from(sourceSets.main.get().allSource)
        }
        
        publishing.publications.withType<MavenPublication>().configureEach {
            artifact(javadocJar)
            artifact(sourcesJar)
        }
        
        // Enable signing for release builds
        if (System.getenv("SIGNING_KEY") != null) {
            apply(plugin = "signing")
            extensions.configure<SigningExtension>("signing") {
                useInMemoryPgpKeys(
                    System.getenv("SIGNING_KEY"),
                    System.getenv("SIGNING_PASSWORD")
                )
                sign(publishing.publications["mavenJava"])
            }
        }
    }
}



================================================
FILE: CONTRIBUTING.md
================================================
# Contributing to Tafkir

Thank you for your interest in contributing to Tafkir! We welcome contributions from the community to help build a robust Java-based ML/AI training framework.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

Please be respectful and constructive in your interactions. We're building an inclusive community for Java developers interested in machine learning.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/tafkir.git
   cd tafkir
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/bhangun/tafkir.git
   ```

## Development Setup

### Prerequisites

- **JDK 25+** (required - Tafkir uses Vector API and Panama FFM)
- **Gradle 8.x** (or use the included Gradle wrapper)
- **Git**

### Build the Project

```bash
# Build and publish to local Maven repository
./gradlew publishToMavenLocal -x test

# Run tests
./gradlew test

# Build specific module
./gradlew :ml:tafkir-ml-api:build
```

### IDE Setup

Tafkir works with any Java IDE:
- **IntelliJ IDEA** (recommended) - Enable preview features for JDK 25
- **Eclipse** - Configure JDK 25 with preview features
- **VS Code** - With Extension Pack for Java

Enable these JVM flags for development:
```
--enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## How to Contribute

### Ways to Contribute

1. **Bug Reports**: Open an issue with detailed reproduction steps
2. **Feature Requests**: Propose new features via GitHub issues
3. **Documentation**: Improve docs, add examples, fix typos
4. **Code Contributions**: Fix bugs or implement features
5. **Testing**: Add test cases or improve test coverage

### Finding Issues to Work On

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Need community assistance
- `bug` - Bugs to fix
- `enhancement` - Features to implement

## Coding Standards

### General Guidelines

- Follow standard Java naming conventions
- Keep methods small and focused (single responsibility)
- Write meaningful comments for complex logic
- Use meaningful variable and method names
- Prefer immutability where possible

### Code Style

```java
// Good: Clear method name, focused responsibility
public Tensor forward(Tensor input) {
    return layers.stream()
        .reduce(input, (acc, layer) -> layer.apply(acc), (a, b) -> b);
}

// Avoid: Vague names, multiple responsibilities
public Tensor f(Tensor t) {
    // ... 100 lines of mixed logic
}
```

### Documentation

- Add Javadoc for public APIs
- Include usage examples in Javadoc
- Update README.md for user-facing changes
- Document breaking changes clearly

Example:
```java
/**
 * Applies a linear transformation to the input tensor.
 * 
 * <p>Computes: output = input * weight^T + bias</p>
 *
 * @param input the input tensor of shape (batch, inFeatures)
 * @return the output tensor of shape (batch, outFeatures)
 * @throws IllegalArgumentException if input dimensions don't match
 */
public Tensor forward(Tensor input);
```

## Testing

### Running Tests

```bash
# Run all tests
./gradlew test

# Run specific module tests
./gradlew :ml:tafkir-ml-api:test

# Run with coverage
./gradlew jacocoTestReport
```

### Writing Tests

- Use JUnit 5 for unit tests
- Use AssertJ for fluent assertions
- Test edge cases and error conditions
- Aim for high test coverage on critical paths

Example:
```java
@Test
void testLinearForwardPass() {
    Linear layer = new Linear(784, 256);
    Tensor input = Tafkir.randn(32, 784);
    
    Tensor output = layer.forward(input);
    
    assertThat(output.shape())
        .containsExactly(32, 256);
    assertThat(output.dtype())
        .isEqualTo(DType.FLOAT32);
}
```

### CI/CD

All pull requests must pass:
- Unit tests
- Integration tests
- Code style checks
- Build verification

The CI pipeline runs automatically on PRs via GitHub Actions.

## Submitting Changes

### Branch Naming

Use descriptive branch names:
```
feature/add-transformer-layer
fix/tensor-shape-validation
docs/update-quickstart
test/add-autograd-tests
```

### Commit Messages

Follow conventional commit format:
```
feat: add RMSNorm layer implementation

Implement RMSNorm (Root Mean Square Layer Normalization) 
as an alternative to LayerNorm for transformer models.

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a PR** with:
   - Clear title describing the change
   - Detailed description of what and why
   - Reference to related issues
   - Screenshots/logs if applicable

3. **Address feedback** from reviewers

4. **Squash commits** if requested before merge

### PR Checklist

Before submitting your PR, ensure:
- [ ] Code follows project style guidelines
- [ ] Tests are added/updated
- [ ] All tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear and conventional
- [ ] Branch is up-to-date with main

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: Check repository maintainers for contact info

### Recognition

Contributors will be:
- Listed in the CONTRIBUTORS file
- Mentioned in release notes for significant contributions
- Recognized in project documentation

---

**Thank you for contributing to Tafkir!** 🎉

Your contributions help make Java a first-class citizen in the ML/AI ecosystem.



================================================
FILE: gradle.properties
================================================
# Tafkir Engine Feature Flags
# Comma-separated list of backends to include: cpu, cuda, metal
tafkir.backend=cpu,metal
# Strategy for backend selection: fallback, strict
tafkir.strategy=fallback
# Enable training APIs (Autograd, ML, Train)
tafkir.training=false
# Enable inference APIs (Runtime, Diffusion, Model)
tafkir.inference=true

# Optimization Flags
org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
org.gradle.parallel=true
org.gradle.caching=true



================================================
FILE: gradlew
================================================
#!/bin/sh

#
# Copyright © 2015-2021 the original authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

##############################################################################
#
#   Gradle start up script for POSIX generated by Gradle.
#
#   Important for running:
#
#   (1) You need a POSIX-compliant shell to run this script. If your /bin/sh is
#       noncompliant, but you have some other compliant shell such as ksh or
#       bash, then to run this script, type that shell name before the whole
#       command line, like:
#
#           ksh Gradle
#
#       Busybox and similar reduced shells will NOT work, because this script
#       requires all of these POSIX shell features:
#         * functions;
#         * expansions «$var», «${var}», «${var:-default}», «${var+SET}»,
#           «${var#prefix}», «${var%suffix}», and «$( cmd )»;
#         * compound commands having a testable exit status, especially «case»;
#         * various built-in commands including «command», «set», and «ulimit».
#
#   Important for patching:
#
#   (2) This script targets any POSIX shell, so it avoids extensions provided
#       by Bash, Ksh, etc; in particular arrays are avoided.
#
#       The "traditional" practice of packing multiple parameters into a
#       space-separated string is a well documented source of bugs and security
#       problems, so this is (mostly) avoided, by progressively accumulating
#       options in "$@", and eventually passing that to Java.
#
#       Where the inherited environment variables (DEFAULT_JVM_OPTS, JAVA_OPTS,
#       and GRADLE_OPTS) rely on word-splitting, this is performed explicitly;
#       see the in-line comments for details.
#
#       There are tweaks for specific operating systems such as AIX, CygWin,
#       Darwin, MinGW, and NonStop.
#
#   (3) This script is generated from the Groovy template
#       https://github.com/gradle/gradle/blob/HEAD/subprojects/plugins/src/main/resources/org/gradle/api/internal/plugins/unixStartScript.txt
#       within the Gradle project.
#
#       You can find Gradle at https://github.com/gradle/gradle/.
#
##############################################################################

# Attempt to set APP_HOME

# Resolve links: $0 may be a link
app_path=$0

# Need this for daisy-chained symlinks.
while
    APP_HOME=${app_path%"${app_path##*/}"}  # leaves a trailing /; empty if no leading path
    [ -h "$app_path" ]
do
    ls=$( ls -ld "$app_path" )
    link=${ls#*' -> '}
    case $link in             #(
      /*)   app_path=$link ;; #(
      *)    app_path=$APP_HOME$link ;;
    esac
done

# This is normally unused
# shellcheck disable=SC2034
APP_BASE_NAME=${0##*/}
# Discard cd standard output in case $CDPATH is set (https://github.com/gradle/gradle/issues/25036)
APP_HOME=$( cd "${APP_HOME:-./}" > /dev/null && pwd -P ) || exit

# Use the maximum available, or set MAX_FD != -1 to use that value.
MAX_FD=maximum

warn () {
    echo "$*"
} >&2

die () {
    echo
    echo "$*"
    echo
    exit 1
} >&2

# OS specific support (must be 'true' or 'false').
cygwin=false
msys=false
darwin=false
nonstop=false
case "$( uname )" in                #(
  CYGWIN* )         cygwin=true  ;; #(
  Darwin* )         darwin=true  ;; #(
  MSYS* | MINGW* )  msys=true    ;; #(
  NONSTOP* )        nonstop=true ;;
esac

CLASSPATH=$APP_HOME/gradle/wrapper/gradle-wrapper.jar


# Determine the Java command to use to start the JVM.
if [ -n "$JAVA_HOME" ] ; then
    if [ -x "$JAVA_HOME/jre/sh/java" ] ; then
        # IBM's JDK on AIX uses strange locations for the executables
        JAVACMD=$JAVA_HOME/jre/sh/java
    else
        JAVACMD=$JAVA_HOME/bin/java
    fi
    if [ ! -x "$JAVACMD" ] ; then
        die "ERROR: JAVA_HOME is set to an invalid directory: $JAVA_HOME

Please set the JAVA_HOME variable in your environment to match the
location of your Java installation."
    fi
else
    JAVACMD=java
    if ! command -v java >/dev/null 2>&1
    then
        die "ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.

Please set the JAVA_HOME variable in your environment to match the
location of your Java installation."
    fi
fi

# Increase the maximum file descriptors if we can.
if ! "$cygwin" && ! "$darwin" && ! "$nonstop" ; then
    case $MAX_FD in #(
      max*)
        # In POSIX sh, ulimit -H is undefined. That's why the result is checked to see if it worked.
        # shellcheck disable=SC2039,SC3045
        MAX_FD=$( ulimit -H -n ) ||
            warn "Could not query maximum file descriptor limit"
    esac
    case $MAX_FD in  #(
      '' | soft) :;; #(
      *)
        # In POSIX sh, ulimit -n is undefined. That's why the result is checked to see if it worked.
        # shellcheck disable=SC2039,SC3045
        ulimit -n "$MAX_FD" ||
            warn "Could not set maximum file descriptor limit to $MAX_FD"
    esac
fi

# Collect all arguments for the java command, stacking in reverse order:
#   * args from the command line
#   * the main class name
#   * -classpath
#   * -D...appname settings
#   * --module-path (only if needed)
#   * DEFAULT_JVM_OPTS, JAVA_OPTS, and GRADLE_OPTS environment variables.

# For Cygwin or MSYS, switch paths to Windows format before running java
if "$cygwin" || "$msys" ; then
    APP_HOME=$( cygpath --path --mixed "$APP_HOME" )
    CLASSPATH=$( cygpath --path --mixed "$CLASSPATH" )

    JAVACMD=$( cygpath --unix "$JAVACMD" )

    # Now convert the arguments - kludge to limit ourselves to /bin/sh
    for arg do
        if
            case $arg in                                #(
              -*)   false ;;                            # don't mess with options #(
              /?*)  t=${arg#/} t=/${t%%/*}              # looks like a POSIX filepath
                    [ -e "$t" ] ;;                      #(
              *)    false ;;
            esac
        then
            arg=$( cygpath --path --ignore --mixed "$arg" )
        fi
        # Roll the args list around exactly as many times as the number of
        # args, so each arg winds up back in the position where it started, but
        # possibly modified.
        #
        # NB: a `for` loop captures its iteration list before it begins, so
        # changing the positional parameters here affects neither the number of
        # iterations, nor the values presented in `arg`.
        shift                   # remove old arg
        set -- "$@" "$arg"      # push replacement arg
    done
fi


# Add default JVM options here. You can also use JAVA_OPTS and GRADLE_OPTS to pass JVM options to this script.
DEFAULT_JVM_OPTS='"-Xmx64m" "-Xms64m"'

# Collect all arguments for the java command:
#   * DEFAULT_JVM_OPTS, JAVA_OPTS, JAVA_OPTS, and optsEnvironmentVar are not allowed to contain shell fragments,
#     and any embedded shellness will be escaped.
#   * For example: A user cannot expect ${Hostname} to be expanded, as it is an environment variable and will be
#     treated as '${Hostname}' itself on the command line.

set -- \
        "-Dorg.gradle.appname=$APP_BASE_NAME" \
        -classpath "$CLASSPATH" \
        org.gradle.wrapper.GradleWrapperMain \
        "$@"

# Stop when "xargs" is not available.
if ! command -v xargs >/dev/null 2>&1
then
    die "xargs is not available"
fi

# Use "xargs" to parse quoted args.
#
# With -n1 it outputs one arg per line, with the quotes and backslashes removed.
#
# In Bash we could simply go:
#
#   readarray ARGS < <( xargs -n1 <<<"$var" ) &&
#   set -- "${ARGS[@]}" "$@"
#
# but POSIX shell has neither arrays nor command substitution, so instead we
# post-process each arg (as a line of input to sed) to backslash-escape any
# character that might be a shell metacharacter, then use eval to reverse
# that process (while maintaining the separation between arguments), and wrap
# the whole thing up as a single "set" statement.
#
# This will of course break if any of these variables contains a newline or
# an unmatched quote.
#

eval "set -- $(
        printf '%s\n' "$DEFAULT_JVM_OPTS $JAVA_OPTS $GRADLE_OPTS" |
        xargs -n1 |
        sed ' s~[^-[:alnum:]+,./:=@_]~\\&~g; ' |
        tr '\n' ' '
    )" '"$@"'

exec "$JAVACMD" "$@"



================================================
FILE: gradlew.bat
================================================
@rem
@rem Copyright 2015 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      https://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem

@if "%DEBUG%"=="" @echo off
@rem ##########################################################################
@rem
@rem  Gradle startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set DIRNAME=%~dp0
if "%DIRNAME%"=="" set DIRNAME=.
@rem This is normally unused
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%

@rem Resolve any "." and ".." in APP_HOME to make it shorter.
for %%i in ("%APP_HOME%") do set APP_HOME=%%~fi

@rem Add default JVM options here. You can also use JAVA_OPTS and GRADLE_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS="-Xmx64m" "-Xms64m"

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if %ERRORLEVEL% equ 0 goto execute

echo. 1>&2
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH. 1>&2
echo. 1>&2
echo Please set the JAVA_HOME variable in your environment to match the 1>&2
echo location of your Java installation. 1>&2

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto execute

echo. 1>&2
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME% 1>&2
echo. 1>&2
echo Please set the JAVA_HOME variable in your environment to match the 1>&2
echo location of your Java installation. 1>&2

goto fail

:execute
@rem Setup the command line

set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar


@rem Execute Gradle
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GRADLE_OPTS% "-Dorg.gradle.appname=%APP_BASE_NAME%" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*

:end
@rem End local scope for the variables with windows NT shell
if %ERRORLEVEL% equ 0 goto mainEnd

:fail
rem Set variable GRADLE_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% equ 0 set EXIT_CODE=1
if not ""=="%GRADLE_EXIT_CONSOLE%" exit %EXIT_CODE%
exit /b %EXIT_CODE%

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega



================================================
FILE: install-local-macos.sh
================================================
#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/install-local-macos.sh"

if [[ ! -x "$SCRIPT_PATH" ]]; then
  echo "Missing $SCRIPT_PATH"
  echo "Run ./create-builder-script.sh to generate it."
  exit 1
fi

exec "$SCRIPT_PATH" "$@"



================================================
FILE: LICENSE.md
================================================
The MIT License (MIT)

Copyright (c) 2026 Bhangun Hartani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


================================================
FILE: Makefile
================================================
.PHONY: error-codes product-boundaries ci

error-codes:
	./scripts/generate-error-codes.sh

product-boundaries:
	./scripts/check-product-boundaries.sh

ci: product-boundaries
	CI=true ./gradlew --no-daemon -x test :spi:tafkir-spi:build



================================================
FILE: Makefile.native
================================================
# Native Library Installation Makefile
# Helps copy native libraries to standard Tafkir location (~/.tafkir/libs/)

.PHONY: help install-native-libs install-gguf-libs install-onnx-libs install-libtorch-libs install-litert-libs clean-native-libs

TAFKIR_LIBS_DIR ?= $(HOME)/.tafkir/libs
LLAMA_LIBS_DIR = $(TAFKIR_LIBS_DIR)/llama
ONNX_LIBS_DIR = $(TAFKIR_LIBS_DIR)/onnxruntime
LIBTORCH_LIBS_DIR = $(TAFKIR_LIBS_DIR)/libtorch
TFLITE_LIBS_DIR = $(TAFKIR_LIBS_DIR)/litert

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = .dylib
    LIB_PREFIX = lib
else ifeq ($(UNAME_S),Linux)
    LIB_EXT = .so
    LIB_PREFIX = lib
else
    # Windows (MinGW/Cygwin)
    LIB_EXT = .dll
    LIB_PREFIX = 
endif

help:
	@echo "Tafkir Native Library Installation"
	@echo "==================================="
	@echo ""
	@echo "Usage:"
	@echo "  make install-native-libs     - Install all native libraries"
	@echo "  make install-gguf-libs       - Install GGUF/llama.cpp libraries"
	@echo "  make install-onnx-libs       - Install ONNX Runtime libraries"
	@echo "  make install-libtorch-libs   - Install LibTorch libraries"
	@echo "  make install-litert-libs     - Install TensorFlow Lite libraries"
	@echo "  make clean-native-libs       - Remove all installed libraries"
	@echo ""
	@echo "Standard location: $(TAFKIR_LIBS_DIR)"
	@echo ""

# Install all native libraries
install-native-libs: install-gguf-libs install-onnx-libs install-libtorch-libs install-litert-libs
	@echo "✓ All native libraries installed to $(TAFKIR_LIBS_DIR)"

# Install GGUF / llama.cpp libraries
install-gguf-libs:
	@echo "Installing GGUF/llama.cpp libraries..."
	@mkdir -p $(LLAMA_LIBS_DIR)
	@# Try multiple source locations
	@if [ -d "plugins/runner/gguf/tafkir-ext-runner-gguf/src/main/resources/native-libs/cpu" ]; then \
		cp plugins/runner/gguf/tafkir-ext-runner-gguf/src/main/resources/native-libs/cpu/*$(LIB_EXT)* $(LLAMA_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -d "extension/format/gguf/tafkir-ext-runner-gguf/target/llama-cpp/lib" ]; then \
		cp extension/format/gguf/tafkir-ext-runner-gguf/target/llama-cpp/lib/*$(LIB_EXT)* $(LLAMA_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -d "plugins/runner/gguf/source/llama-cpp/lib/cpu" ]; then \
		cp plugins/runner/gguf/source/llama-cpp/lib/cpu/*$(LIB_EXT)* $(LLAMA_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -d "extension/format/gguf/vendor/llama-cpp/lib/cpu" ]; then \
		cp extension/format/gguf/vendor/llama-cpp/lib/cpu/*$(LIB_EXT)* $(LLAMA_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -d "vendor/llama-cpp/lib/cpu" ]; then \
		cp vendor/llama-cpp/lib/cpu/*$(LIB_EXT)* $(LLAMA_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@# Set executable permissions
	@chmod +x $(LLAMA_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || true
	@# Clear macOS quarantine
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
		xattr -dr com.apple.quarantine $(LLAMA_LIBS_DIR) 2>/dev/null || true; \
	fi
	@echo "✓ GGUF libraries installed to $(LLAMA_LIBS_DIR)"
	@ls -lh $(LLAMA_LIBS_DIR)/ 2>/dev/null || echo "  (no libraries found)"

# Install ONNX Runtime libraries
install-onnx-libs:
	@echo "Installing ONNX Runtime libraries..."
	@mkdir -p $(ONNX_LIBS_DIR)
	@# Try multiple source locations
	@if [ -d "plugins/runner/onnx/tafkir-runner-onnx/src/main/cpp/onnxruntime/build/onnxruntime-osx-arm64-1.19.2/lib" ]; then \
		cp plugins/runner/onnx/tafkir-runner-onnx/src/main/cpp/onnxruntime/build/onnxruntime-osx-arm64-1.19.2/lib/*$(LIB_EXT)* $(ONNX_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -d "inference/format/onnx/onnxruntime/build/lib" ]; then \
		cp inference/format/onnx/onnxruntime/build/lib/*$(LIB_EXT)* $(ONNX_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@# Set executable permissions
	@chmod +x $(ONNX_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || true
	@# Clear macOS quarantine
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
		xattr -dr com.apple.quarantine $(ONNX_LIBS_DIR) 2>/dev/null || true; \
	fi
	@echo "✓ ONNX Runtime libraries installed to $(ONNX_LIBS_DIR)"
	@ls -lh $(ONNX_LIBS_DIR)/ 2>/dev/null || echo "  (no libraries found)"

# Install LibTorch libraries
install-libtorch-libs:
	@echo "Installing LibTorch libraries..."
	@mkdir -p $(LIBTORCH_LIBS_DIR)
	@# Try multiple source locations
	@if [ -d "extension/kernel/libtorch/build/lib" ]; then \
		cp extension/kernel/libtorch/build/lib/*$(LIB_EXT)* $(LIBTORCH_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -d "plugins/kernel/libtorch/build/lib" ]; then \
		cp plugins/kernel/libtorch/build/lib/*$(LIB_EXT)* $(LIBTORCH_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -d "extension/kernel/libtorch/src/main/resources/native/Darwin/arm64" ]; then \
		cp extension/kernel/libtorch/src/main/resources/native/Darwin/arm64/*$(LIB_EXT)* $(LIBTORCH_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@# Set executable permissions
	@chmod +x $(LIBTORCH_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || true
	@# Clear macOS quarantine
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
		xattr -dr com.apple.quarantine $(LIBTORCH_LIBS_DIR) 2>/dev/null || true; \
	fi
	@echo "✓ LibTorch libraries installed to $(LIBTORCH_LIBS_DIR)"
	@ls -lh $(LIBTORCH_LIBS_DIR)/ 2>/dev/null || echo "  (no libraries found)"

# Install TensorFlow Lite libraries
install-litert-libs:
	@echo "Installing TensorFlow Lite libraries..."
	@mkdir -p $(TFLITE_LIBS_DIR)
	@# Try multiple source locations
	@if [ -d "plugins/runner/litert/tafkir-runner-litert/src/main/resources/native-libs" ]; then \
		cp plugins/runner/litert/tafkir-runner-litert/src/main/resources/native-libs/*$(LIB_EXT)* $(TFLITE_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -d "extension/runner/litert/build/lib" ]; then \
		cp extension/runner/litert/build/lib/*$(LIB_EXT)* $(TFLITE_LIBS_DIR)/ 2>/dev/null || true; \
	fi
	@# Set executable permissions
	@chmod +x $(TFLITE_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || true
	@# Clear macOS quarantine
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
		xattr -dr com.apple.quarantine $(TFLITE_LIBS_DIR) 2>/dev/null || true; \
	fi
	@echo "✓ TensorFlow Lite libraries installed to $(TFLITE_LIBS_DIR)"
	@ls -lh $(TFLITE_LIBS_DIR)/ 2>/dev/null || echo "  (no libraries found)"

# Clean all installed libraries
clean-native-libs:
	@echo "Removing all installed native libraries..."
	@rm -rf $(TAFKIR_LIBS_DIR)/llama
	@rm -rf $(TAFKIR_LIBS_DIR)/onnxruntime
	@rm -rf $(TAFKIR_LIBS_DIR)/libtorch
	@rm -rf $(TAFKIR_LIBS_DIR)/litert
	@echo "✓ All native libraries removed from $(TAFKIR_LIBS_DIR)"

# Verify installation
verify-libs:
	@echo "Verifying native library installation..."
	@echo ""
	@echo "GGUF/llama.cpp:"
	@ls -lh $(LLAMA_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || echo "  Not installed"
	@echo ""
	@echo "ONNX Runtime:"
	@ls -lh $(ONNX_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || echo "  Not installed"
	@echo ""
	@echo "LibTorch:"
	@ls -lh $(LIBTORCH_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || echo "  Not installed"
	@echo ""
	@echo "TensorFlow Lite:"
	@ls -lh $(TFLITE_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || echo "  Not installed"
	@echo ""

# Download pre-built binaries (optional)
download-gguf:
	@echo "Downloading pre-built llama.cpp binaries..."
	@mkdir -p $(LLAMA_LIBS_DIR)
	@curl -L https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-bin-macos-arm64.tar.gz | tar xz -C $(LLAMA_LIBS_DIR) || echo "Download failed"
	@chmod +x $(LLAMA_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || true

download-onnx:
	@echo "Downloading pre-built ONNX Runtime..."
	@mkdir -p $(ONNX_LIBS_DIR)
	@curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-arm64-1.19.2.tgz -o /tmp/onnxruntime.tgz
	@tar xzf /tmp/onnxruntime.tgz -C $(ONNX_LIBS_DIR) --strip-components=1
	@cp $(ONNX_LIBS_DIR)/lib/*$(LIB_EXT)* $(ONNX_LIBS_DIR)/ 2>/dev/null || true
	@rm -rf $(ONNX_LIBS_DIR)/lib
	@rm /tmp/onnxruntime.tgz
	@chmod +x $(ONNX_LIBS_DIR)/*$(LIB_EXT)* 2>/dev/null || true



================================================
FILE: QUICKSTART.md
================================================
# Tafkir Quickstart

Get started with Tafkir ML framework in 5 minutes.

---

## 1. Prerequisites

- **JDK 25+** (required for Vector API and Panama FFM features)
- **JBang** (optional, for scripting examples)

---

## 2. Install

### Option A: JBang (Recommended - No Build Tools Needed)

```bash
curl -Ls https://sh.jbang.dev | bash -s -
```

### Option B: Gradle (Build from Source)

```bash
git clone https://github.com/bhangun/tafkir.git
cd tafkir
./gradlew publishToMavenLocal -x test
```

### Option C: Maven Dependency (Once Published)

```xml
<dependency>
    <groupId>tech.kayys.tafkir</groupId>
    <artifactId>tafkir-ml-api</artifactId>
    <version>0.3.0</version>
</dependency>
```

---

## 3. Verify Installation

```bash
# Via JBang (from examples directory)
cd examples/jbang
jbang sdk/tafkir-quickstart.java

# Or run any JBang example
jbang common/hello_tafkir.java
```

Expected output:
```
✓ Tafkir SDK installed successfully
✓ Tensor operations working
```

---

## 4. Create Your First Model

### Using JBang

Create `my_model.java`:

```java
///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.tafkir:tafkir-ml-api:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.Tafkir;
import tech.kayys.tafkir.ml.tensor.Tensor;
import tech.kayys.tafkir.ml.nn.Linear;
import tech.kayys.tafkir.ml.nn.ReLU;
import tech.kayys.tafkir.ml.nn.Sequential;

public class my_model {
    public static void main(String[] args) {
        // Build a simple neural network
        Sequential model = Sequential.builder()
            .add(new Linear(784, 256))
            .add(new ReLU())
            .add(new Linear(256, 10))
            .build();

        // Create random input (simulating a 28x28 image)
        Tensor input = Tafkir.randn(1, 784);

        // Forward pass
        Tensor output = model.forward(input);

        System.out.println("Model: 784 → 256 → 10");
        System.out.println("Input shape: [1, 784]");
        System.out.println("Output shape: " + java.util.Arrays.toString(output.shape()));
    }
}
```

Run:
```bash
jbang my_model.java
```

---

## 5. Run Inference

```java
import tech.kayys.tafkir.ml.runner.*;

ModelRunner runner = ModelRunner.builder()
    .modelPath("model.onnx")
    .device(RunnerDevice.CUDA)
    .build();

InferenceResult result = runner.infer(
    InferenceInput.fromFloats(inputData, 1, 3, 224, 224)
);

System.out.println("Output: " + result.primaryOutput().asFloats());
System.out.println("Latency: " + result.latency());

runner.close();
```

---

## 6. Explore Examples

```bash
# Browse all examples
ls examples/jbang/

# Run comprehensive demo
jbang examples/jbang/sdk/unified_framework_demo.java --demo all

# Open Jupyter notebooks
cd examples/jupyter/
jupyter notebook
```

---

## Next Steps

- **[Examples Setup](examples/docs/SETUP.md)** - Detailed example setup guide
- **[Framework Guide](website/tafkir-ai.github.io/docs/framework/)** - ML framework docs
- **[Inference Guide](website/tafkir-ai.github.io/docs/framework/inference.md)** - Model inference docs
- **[CLI Documentation](tafkir-cli/README.md)** - Command-line tool reference
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to Tafkir



================================================
FILE: RELEASE_GUIDE.md
================================================
# Tafkir Release Guide

This document describes how to create and publish a new release of Tafkir.

## Prerequisites

- JDK 25+ installed
- Access to the `bhangun/tafkir` GitHub repository with write permissions
- Maven Central account (for publishing to Maven Central)
- GPG key configured for signing artifacts

## Version Numbering

Tafkir follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 0.3.0)
- **SNAPSHOT** versions for development (e.g., 0.3.0-SNAPSHOT)

### When to increment:

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

## Release Checklist

### 1. Prepare the Release

- [ ] Ensure all tests pass: `./gradlew test`
- [ ] Update version in `build.gradle.kts` from SNAPSHOT to release version
- [ ] Update `QUICKSTART.md` with any new features or changes
- [ ] Update `README.md` if needed
- [ ] Review and merge all pending PRs
- [ ] Create a release branch (optional for major releases): `git checkout -b release/v0.3.0`

### 2. Test the Build

```bash
# Clean build
./gradlew clean build

# Run all tests
./gradlew test

# Build Javadoc
./gradlew javadocJar

# Build sources jar
./gradlew sourcesJar
```

### 3. Create Git Tag

```bash
# Commit version changes
git add .
git commit -m "Release v0.3.0"

# Create annotated tag
git tag -a v0.3.0 -m "Tafkir v0.3.0"

# Push tag to GitHub
git push origin v0.3.0
```

### 4. Trigger GitHub Release Workflow

The `.github/workflows/release.yml` workflow will automatically:

1. Validate the version
2. Build and test the project
3. Generate Javadoc and sources jars
4. Build native CLI executables for Linux, macOS, and Windows
5. Build JVM package
6. Create GitHub Release with all assets
7. Publish to Maven Central (if configured)

### Manual Release (Workflow Dispatch)

If you need to trigger a release manually:

1. Go to **Actions** → **Release Tafkir**
2. Click **Run workflow**
3. Enter the version number (e.g., `0.3.0`)
4. Click **Run workflow**

### 5. Verify Release

After the workflow completes:

- [ ] Check [GitHub Releases](https://github.com/bhangun/tafkir/releases) for the new release
- [ ] Verify all assets are uploaded:
  - Native binaries (Linux, macOS, Windows)
  - JVM package
  - Javadoc jars
- [ ] Check [Maven Central](https://central.sonatype.com/) for published artifacts (may take 30 minutes to appear)
- [ ] Test installation via JBang: `jbang tafkir@bhangun/tafkir`

### 6. Post-Release Tasks

- [ ] Update version back to SNAPSHOT: `0.4.0-SNAPSHOT`
- [ ] Create release announcement (blog post, social media, etc.)
- [ ] Update documentation website if applicable
- [ ] Notify community (Discord, mailing list, etc.)

## Publishing to Maven Central

### One-Time Setup

1. **Create Sonatype Account**: Register at [OSSRH](https://issues.sonatype.org/)
2. **Create JIRA Ticket**: Request namespace `tech.kayys`
3. **Generate GPG Key**:
   ```bash
   gpg --gen-key
   gpg --armor --export your-email@example.com
   ```
4. **Configure GitHub Secrets**:
   - `MAVEN_CENTRAL_USERNAME`: Your Sonatype username
   - `MAVEN_CENTRAL_PASSWORD`: Your Sonatype password/token
   - `SIGNING_KEY`: Your GPG private key (armored)
   - `SIGNING_PASSWORD`: Your GPG passphrase

### Manual Publishing

```bash
export ORG_GRADLE_PROJECT_mavenCentralUsername=your-username
export ORG_GRADLE_PROJECT_mavenCentralPassword=your-password
export ORG_GRADLE_PROJECT_signingInMemoryKey=$(cat ~/.gnupg/private.key)
export ORG_GRADLE_PROJECT_signingInMemoryKeyPassword=your-passphrase

./gradlew publishToMavenCentral
```

## Troubleshooting

### Build Fails

Check the GitHub Actions logs for specific errors. Common issues:

- **Java version mismatch**: Ensure JDK 25 is used
- **Missing dependencies**: Run `./gradlew dependencies` to diagnose
- **Test failures**: Fix failing tests before release

### Maven Central Publishing Fails

- **Signature validation failed**: Verify GPG key is correctly configured
- **Invalid POM**: Check that all required metadata is present
- **Repository closed**: Wait for Sonatype review (usually 2 hours)

### Native Build Fails

- **Linux**: Install `build-essential`, `libssl-dev`, `zlib1g-dev`
- **macOS**: Install Xcode command line tools and OpenSSL via Homebrew
- **Windows**: Ensure Visual Studio Build Tools are installed

## Emergency Hotfix

For critical bug fixes:

1. Create hotfix branch from release tag: `git checkout -b hotfix/v0.3.1 v0.3.0`
2. Apply fix and update version to `0.3.1`
3. Follow normal release process
4. Merge changes back to main branch

## Contact

For questions about the release process, open an issue or contact the maintainers.



================================================
FILE: settings.gradle.kts
================================================
rootProject.name = "tafkir-engine"

includeBuild("../aljabr")

fun includeOptionalProject(projectPath: String, vararg candidatePaths: String) {
    val projectDir = candidatePaths
        .map { file(it) }
        .firstOrNull { candidate ->
            candidate.resolve("build.gradle.kts").isFile || candidate.resolve("build.gradle").isFile
        }
        ?: return

    include(projectPath)
    project(":$projectPath").projectDir = projectDir
}

val staticallyIncludedModelProjects = setOf<String>()

includeOptionalProject("suling", "../extensions/audio/suling", "stubs/suling")

include("quantizer:tafkir-quantizer-autoround")
include("quantizer:tafkir-quantizer-awq")
include("quantizer:tafkir-quantizer-gptq")
include("quantizer:tafkir-quantizer-quip")
include("quantizer:tafkir-quantizer-turboquant")

if (file("sdk/tafkir-sdk-session").isDirectory) {
//    include("sdk:tafkir-sdk-session")
}

include("ml:tafkir-ml-core")
include("ml:tafkir-ml-autograd")
include("ml:tafkir-ml-timeseries")


includeOptionalProject("training:tafkir-train-strategy", "training/tafkir-train-strategy")
include("integration:jupyter:tafkir-jupyter-kernel")



================================================
FILE: TAFKIR_LEARN_STRATEGY.md
================================================
# Tafkir-Learn: Strategic Architecture & Roadmap

## Vision
Position **Tafkir-Learn** as the premier JVM-native deep learning framework, serving as a direct alternative to PyTorch, TensorFlow, and JAX while leveraging the Aljabr compute engine and integrating with Aqli for production inference.

---

## Three-Pillar Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│              (User Code, Examples, Notebooks)               │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌─────────────────┐  ┌───────────────┐
│   Tafkir      │  │    Tafkir       │  │    Aqli     │
│   Learn       │  │    Hub          │  │   Inference   │
│  (Training)   │  │  (Model Zoo)    │  │   Engine      │
│               │  │                 │  │               │
│ • nn.Module   │  │ • Pre-trained   │  │ • Graph Opt   │
│ • Optimizers  │  │   Models        │  │ • Quantization│
│ • DataLoader  │  │ • Model Cards   │  │ • Serving     │
│ • Trainer API │  │ • Versioning    │  │ • Runtime     │
│ • Metrics     │  │                 │  │               │
└───────┬───────┘  └────────┬────────┘  └───────▲───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │     Aljabr      │
                  │  (Compute Core) │
                  │                 │
                  │ • Tensor Ops    │
                  │ • Autograd      │
                  │ • GPU/CPU Back- │
                  │   ends          │
                  │ • JIT Compiler  │
                  └─────────────────┘
```

---

## Responsibility Boundaries

### **Aljabr** (Backend - `github.com/bhangun/aljabr`)
**Role:** Pure computational engine (equivalent to CUDA + cuDNN + ATen)

**Should Contain:**
- ✅ Tensor primitives (`Tensor`, `GradTensor`)
- ✅ Memory management (allocators, pools)
- ✅ BLAS/LAPACK wrappers (SGEMM, DGEMM, etc.)
- ✅ Convolution kernels (im2col, winograd)
- ✅ Attention kernels (flash attention variants)
- ✅ Automatic differentiation core (computation graph, tape)
- ✅ Hardware backends (CPU SIMD, CUDA, Metal, ROCm)
- ✅ JIT compilation infrastructure
- ✅ Basic distributed primitives (NCCL wrappers)

**Should NOT Contain:**
- ❌ Neural network layers (`Linear`, `Conv2d`)
- ❌ Optimizers (`Adam`, `SGD`)
- ❌ Loss functions
- ❌ Training loops
- ❌ Data loading utilities
- ❌ Model architectures

---

### **Tafkir-Learn** (Training Framework - This Project)
**Role:** High-level ML framework (equivalent to PyTorch `torch.nn` + `torch.optim`)

**Current Location:** `/workspace/training/` and `/workspace/trainer/`

**Should Contain:**

#### 1. **Neural Network Modules** (`tafkir-nn`)
```java
// PyTorch-like API
Module model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Dropout(0.5),
    new Linear(256, 10)
);

// Functional API
Tensor output = nn.functional.linear(input, weight, bias);
```

**Submodules:**
- `tafkir-nn` - Core modules (Linear, Conv2d, BatchNorm, etc.)
- `tafkir-nn-cnn` - CNN-specific layers (Conv2d, MaxPool2d, ResNet blocks)
- `tafkir-nn-rnn` - RNN/LSTM/GRU cells
- `tafkir-nn-transformer` - Attention, TransformerEncoder, positional embeddings
- `tafkir-nn-vision` - ViT, detection heads, segmentation
- `tafkir-nn-audio` - Spectrograms, audio preprocessing
- `tafkir-nn-multimodal` - CLIP-style encoders, fusion layers

#### 2. **Optimization** (`tafkir-nn-optimize`)
```java
Optimizer optimizer = new AdamW(model.parameters(), 1e-3)
    .weightDecay(0.01)
    .betas(0.9, 0.999);

Scheduler scheduler = new CosineAnnealingWarmRestartsLR(optimizer, 10);
```

**Includes:**
- All optimizers (SGD, Adam, AdamW, Lion, LAMB, etc.)
- Learning rate schedulers (StepLR, CosineAnnealing, ReduceLROnPlateau)
- Gradient clipping and scaling
- Advanced techniques (Lookahead, SAM, QAT)

#### 3. **Data Pipeline** (`tafkir-train-data`)
```java
Dataset dataset = new MNIST("./data", train=true, transform=ToTensor());
DataLoader loader = new DataLoader(dataset)
    .batchSize(32)
    .shuffle(true)
    .numWorkers(4);
```

**Includes:**
- Dataset base classes and common datasets (MNIST, CIFAR, ImageNet)
- DataLoader with batching, shuffling, parallel loading
- Transforms (resize, normalize, augmentation)
- Samplers (weighted, distributed)

#### 4. **Training API** (`tafkir-train-api`)
```java
Trainer trainer = Trainer.builder(model)
    .optimizer(optimizer)
    .lossFunction(CrossEntropyLoss())
    .metrics(Accuracy(), F1Score())
    .callbacks(EarlyStopping(patience=5), CheckpointSaver())
    .build();

TrainingSummary summary = trainer.fit(trainLoader, valLoader, epochs=10);
```

**Includes:**
- CanonicalTrainer with full training loop
- Metrics (accuracy, precision, recall, F1, AUROC, etc.)
- Callbacks (early stopping, checkpointing, logging)
- Mixed precision training (GradScaler)
- Distributed training strategies (DDP, DeepSpeed integration)

#### 5. **Export & Serialization** (`tafkir-serializer`, `tafkir-ml-export`)
```java
// Export to Aqli format
model.export("model.aqli", format=ExportFormat.AQLI);

// ONNX export
model.export("model.onnx", format=ExportFormat.ONNX);

// TorchScript-like tracing
Graph graph = Tracer.trace(model, exampleInput);
```

#### 6. **Model Hub Integration** (`tafkir-ml-hub`)
```java
// Load pre-trained models
Model model = TafkirHub.load("bert-base-uncased");
Model model = TafkirHub.load("resnet50", pretrained=true);

// Push custom models
TafkirHub.push(myModel, "my-org/my-model");
```

---

### **Aqli** (Inference Engine - `github.com/bhangun/aqli`)
**Role:** Production deployment (equivalent to TensorRT + TorchServe)

**Should Contain:**
- ✅ Model graph optimization (fusion, constant folding)
- ✅ Quantization (FP16, INT8, calibration)
- ✅ Optimized runtime (forward pass only, no autograd)
- ✅ Serving infrastructure (HTTP/gRPC, batching)
- ✅ Multi-model orchestration
- ✅ Hardware-specific optimizations (TensorRT, OpenVINO backends)
- ✅ Model format serialization (`.aqli` format)

**Should NOT Contain:**
- ❌ Training logic
- ❌ Backward pass
- ❌ Optimizers
- ❌ Data augmentation

---

## Renaming Strategy: `tafkir-sklearn` → `tafkir-learn`

### Current Structure Issues
The current codebase has training components scattered across:
- `/workspace/training/` - Main training modules
- `/workspace/trainer/` - Legacy trainer implementations
- `/workspace/ml/` - ML integration layers

### Proposed Module Restructure

```
tafkir-engine/
├── tafkir-learn/                    # NEW: Top-level training framework
│   ├── tafkir-learn-core/           # Core APIs (Module, Parameter, functional)
│   ├── tafkir-learn-nn/             # Neural network layers (renamed from tafkir-nn)
│   ├── tafkir-learn-cnn/            # CNN modules
│   ├── tafkir-learn-rnn/            # RNN modules  
│   ├── tafkir-learn-transformer/    # Transformer modules
│   ├── tafkir-learn-vision/         # Vision-specific
│   ├── tafkir-learn-audio/          # Audio-specific
│   ├── tafkir-learn-multimodal/     # Multimodal
│   ├── tafkir-learn-optim/          # Optimizers & schedulers (renamed from tafkir-nn-optimize)
│   ├── tafkir-learn-data/           # DataLoader & datasets (renamed from tafkir-train-data)
│   ├── tafkir-learn-trainer/        # Training loop & metrics (renamed from tafkir-train-api)
│   ├── tafkir-learn-export/         # Export to ONNX/Aqli
│   ├── tafkir-learn-hub/            # Model hub client
│   └── tafkir-learn-examples/       # Tutorials and examples
│
├── tafkir-aqli-integration/       # Bridge to Aqli inference
├── tafkir-aljabr-integration/       # Bridge to Aljabr backend
│
└── [models/]                        # Keep existing model implementations
    ├── tafkir-model-bert/
    ├── tafkir-model-llama/
    └── ...
```

### Migration Steps

1. **Create `tafkir-learn` parent module**
   - New directory: `/workspace/tafkir-learn/`
   - Aggregator POM/build.gradle.kts

2. **Rename and migrate existing modules**
   ```bash
   mv training/tafkir-nn tafkir-learn/tafkir-learn-nn
   mv training/tafkir-nn-optimize tafkir-learn/tafkir-learn-optim
   mv training/tafkir-train-data tafkir-learn/tafkir-learn-data
   mv training/tafkir-train-api tafkir-learn/tafkir-learn-trainer
   mv training/tafkir-nn-cnn tafkir-learn/tafkir-learn-cnn
   mv training/tafkir-nn-rnn tafkir-learn/tafkir-learn-rnn
   mv training/tafkir-nn-transformer tafkir-learn/tafkir-learn-transformer
   mv training/tafkir-serializer tafkir-learn/tafkir-learn-export
   ```

3. **Update package names**
   - `tech.kayys.tafkir.nn.*` → `tech.kayys.tafkir.learn.nn.*`
   - `tech.kayys.tafkir.ml.train.*` → `tech.kayys.tafkir.learn.trainer.*`
   - `tech.kayys.tafkir.nn.optimize.*` → `tech.kayys.tafkir.learn.optim.*`

4. **Update dependencies in settings.gradle.kts**
   - Replace old module references with new `tafkir-learn:*` paths

5. **Deprecate old modules gradually**
   - Keep old module names as thin wrappers for one release cycle
   - Add `@Deprecated` annotations with migration guidance

---

## Key Differentiators vs PyTorch

### What Tafkir-Learn Does Better

1. **Type Safety**
   - Kotlin/Java static typing catches errors at compile time
   - No runtime tensor shape surprises (with proper type hints)

2. **JVM Ecosystem Integration**
   - Direct integration with Spring Boot, Quarkus, Micronaut
   - Seamless deployment in enterprise Java environments
   - Access to Java's mature monitoring (JMX, Micrometer)

3. **No Python Dependencies**
   - Pure JVM stack simplifies deployment
   - No conda/virtualenv hell
   - Smaller container images

4. **Performance**
   - Aljabr's Vector API SIMD on modern CPUs
   - Direct GPU access without Python overhead
   - GraalVM native image support for low-latency serving

5. **Production-Ready by Default**
   - Built-in checkpointing with integrity checks
   - Structured training reports (JSON/CSV)
   - Gradient diagnostics and non-finite guards
   - Throughput metrics out of the box

### What Needs Improvement to Match PyTorch

1. **API Completeness**
   - [ ] Implement missing layers (InstanceNorm, GroupNorm variants)
   - [ ] Add more activation functions (Swish, Mish variants)
   - [ ] Complete torchvision equivalents (transforms, models)

2. **Developer Experience**
   - [ ] Functional API parity (`nn.functional.*`)
   - [ ] Method chaining for tensor operations
   - [ ] Better error messages with suggestions

3. **Ecosystem**
   - [ ] Hugging Face Transformers integration
   - [ ] Weights & Biases, MLflow logging
   - [ ] Pre-trained model zoo expansion

4. **Documentation**
   - [ ] Comprehensive tutorials (getting started, migration guides)
   - [ ] API reference documentation (KDoc → docs site)
   - [ ] Performance benchmarks vs PyTorch

5. **Advanced Features**
   - [ ] Distributed training (DDP, FSDP equivalents)
   - [ ] Gradient checkpointing for memory efficiency
   - [ ] AMP (automatic mixed precision) improvements
   - [ ] TorchScript/JIT compilation for optimization

---

## Immediate Action Items

### Phase 1: Foundation (Weeks 1-4)
- [ ] Rename modules to `tafkir-learn-*` convention
- [ ] Consolidate `/trainer/` and `/training/` into unified structure
- [ ] Create cohesive `TafkirLearn` facade class
- [ ] Document Aljabr ↔ Tafkir-Learn boundary clearly

### Phase 2: API Polish (Weeks 5-8)
- [ ] Implement functional API (`nn.functional.*`)
- [ ] Add method chaining to tensor operations
- [ ] Improve error messages with actionable feedback
- [ ] Create PyTorch migration guide

### Phase 3: Ecosystem (Weeks 9-12)
- [ ] Build Hugging Face model loader
- [ ] Integrate W&B/MLflow logging
- [ ] Expand pre-trained model examples
- [ ] Create example repository with notebooks

### Phase 4: Production Features (Weeks 13-16)
- [ ] Complete distributed training support
- [ ] Optimize Aqli export pipeline
- [ ] Add quantization-aware training
- [ ] Benchmark against PyTorch on standard tasks

---

## Marketing Positioning

### Tagline Options
- "PyTorch for the JVM"
- "Deep Learning, Type-Safe"
- "Train on Aljabr, Deploy with Aqli"
- "The JVM's Answer to PyTorch"

### Target Audience
1. **Enterprise Java Teams** - Already using JVM, want ML without Python
2. **Scala/Kotlin Developers** - Appreciate type safety and functional patterns
3. **Production Engineers** - Need reliable, monitorable ML pipelines
4. **Research Engineers** - Want reproducible experiments with versioned dependencies

### Key Messages
- "No Python required - pure JVM from research to production"
- "Compile-time safety meets GPU acceleration"
- "Seamless integration with your existing Java stack"
- "Backed by Aljabr's high-performance compute engine"

---

## Success Metrics

### Technical
- [ ] 90%+ PyTorch API coverage for core modules
- [ ] Within 20% of PyTorch training performance on CPU
- [ ] Within 10% of PyTorch training performance on GPU
- [ ] < 1ms overhead vs raw Aljabr for forward pass

### Adoption
- [ ] 100+ GitHub stars in first 3 months post-rebrand
- [ ] 10+ external contributors
- [ ] 5+ production case studies
- [ ] Integration with major JVM frameworks (Spring, Quarkus)

### Community
- [ ] Active Discord/Slack community (500+ members)
- [ ] Monthly contributor calls
- [ ] Regular blog posts and tutorials
- [ ] Presence at JVM conferences (Devoxx, JavaOne)

---

## Conclusion

Renaming to **Tafkir-Learn** clarifies the project's purpose as a comprehensive training framework while maintaining clear boundaries with Aljabr (compute) and Aqli (inference). This three-pillar architecture mirrors successful patterns from PyTorch (torch + torchserve) and TensorFlow (TF + TFLite + TF Serving), positioning Tafkir as a production-ready, JVM-native alternative for the entire ML lifecycle.



================================================
FILE: buildSrc/build.gradle.kts
================================================
plugins {
    `kotlin-dsl`
}

repositories {
    gradlePluginPortal()
    mavenCentral()
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverage.kt
================================================
package tech.kayys.tafkir.buildlogic

data class DirectFixtureCoverage(
    val id: String,
    val modelType: String,
    val architectures: List<String>,
    val tokenizerMarker: String,
)



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageFileSupport.kt
================================================
package tech.kayys.tafkir.buildlogic

import java.io.File

object DirectFixtureCoverageFileSupport {
    fun fixtureModelType(config: Map<*, *>): String =
        config["model_type"]?.toString()?.trim().orEmpty()

    fun fixtureArchitectures(config: Map<*, *>): List<String> =
        (config["architectures"] as? List<*>)
            ?.mapNotNull { it?.toString()?.trim() }
            ?.filter { it.isNotBlank() }
            ?: emptyList()

    fun containsJavaSourceText(module: File, relativeDir: String, text: String): Boolean {
        val sourceDir = sourceDir(module, relativeDir)
        if (sourceDir == null) {
            return false
        }

        return containsJavaSourceText(javaSourceFiles(sourceDir), text)
    }

    fun javaSourceTextIfContains(module: File, relativeDir: String, text: String): String? {
        val sourceDir = sourceDir(module, relativeDir)
        if (sourceDir == null) {
            return null
        }

        val sourceFiles = javaSourceFiles(sourceDir)
        if (!containsJavaSourceText(sourceFiles, text)) {
            return null
        }

        return javaSourceText(sourceFiles)
    }

    private fun sourceDir(module: File, relativeDir: String): File? =
        module.resolve(relativeDir).takeIf { it.isDirectory }

    private fun javaSourceFiles(sourceDir: File): List<File> =
        sourceDir.walkTopDown()
            .filter { it.isFile && it.extension == "java" }
            .sortedBy { it.relativeTo(sourceDir).path }
            .toList()

    private fun containsJavaSourceText(sourceFiles: List<File>, text: String): Boolean =
        sourceFiles.any { sourceFile ->
            sourceFile.useLines { lines ->
                lines.any { it.contains(text) }
            }
        }

    private fun javaSourceText(sourceFiles: List<File>): String {
        val sourceText = StringBuilder()
        sourceFiles.forEach { sourceFile ->
            if (sourceText.isNotEmpty()) {
                sourceText.append('\n')
            }
            sourceText.append(sourceFile.readText())
        }
        return sourceText.toString()
    }
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageFixtureParser.kt
================================================
package tech.kayys.tafkir.buildlogic

import groovy.json.JsonSlurper
import java.io.File

object DirectFixtureCoverageFixtureParser {
    fun parse(
        module: File,
        mainJavaText: String,
        config: File,
        problems: MutableList<String>,
    ): DirectFixtureCoverage? {
        val fixturePath = config.relativeTo(module).path
        val parsed = parseConfig(module, fixturePath, config, problems)
        if (parsed != null) {
            validateClaims(module, mainJavaText, fixturePath, parsed, problems)
        }

        val dir = config.parentFile
        val tokenizerMarker = DirectFixtureCoverageReportSpec.tokenizerMarkers.firstOrNull {
            dir.resolve(it).isFile
        }
        if (tokenizerMarker == null) {
            problems += "${module.name}: ${dir.relativeTo(module).path} should include tokenizer.json or vocab marker"
        }
        return parsed?.let {
            DirectFixtureCoverage(
                id = dir.name,
                modelType = DirectFixtureCoverageFileSupport.fixtureModelType(it),
                architectures = DirectFixtureCoverageFileSupport.fixtureArchitectures(it),
                tokenizerMarker = tokenizerMarker.orEmpty(),
            )
        }
    }

    private fun parseConfig(
        module: File,
        fixturePath: String,
        config: File,
        problems: MutableList<String>,
    ): Map<*, *>? =
        try {
            JsonSlurper().parse(config) as? Map<*, *>
        } catch (error: Exception) {
            problems += "${module.name}: $fixturePath is not parseable JSON (${error.javaClass.simpleName})"
            null
        }

    private fun validateClaims(
        module: File,
        mainJavaText: String,
        fixturePath: String,
        parsed: Map<*, *>,
        problems: MutableList<String>,
    ): Unit {
        val modelType = DirectFixtureCoverageFileSupport.fixtureModelType(parsed)
        if (modelType.isBlank()) {
            problems += "${module.name}: $fixturePath should declare model_type"
        } else if (!mainJavaText.contains("\"$modelType\"")) {
            problems += "${module.name}: $fixturePath model_type '$modelType' is not claimed in main Java sources"
        }
        val architectures = parsed["architectures"] as? List<*>
        if (architectures.isNullOrEmpty()) {
            problems += "${module.name}: $fixturePath should declare at least one architecture"
        } else {
            val architectureClaims = DirectFixtureCoverageFileSupport.fixtureArchitectures(parsed)
            if (architectureClaims.isEmpty()) {
                problems += "${module.name}: $fixturePath should declare at least one non-blank architecture"
            }
            architectureClaims.forEach { architecture ->
                if (!mainJavaText.contains("\"$architecture\"")) {
                    problems += "${module.name}: $fixturePath architecture '$architecture' is not claimed in main Java sources"
                }
            }
        }
    }
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageGuards.kt
================================================
package tech.kayys.tafkir.buildlogic

import org.gradle.api.GradleException

object DirectFixtureCoverageGuards {
    fun requireProblemsClear(problems: List<String>): Unit {
        if (problems.isNotEmpty()) {
            throw GradleException(buildString {
                appendLine("Direct model-family fixture coverage problems:")
                append(problems.joinToString(separator = "\n") { "- $it" })
            })
        }
    }

    fun requireReportEntries(
        directModuleCount: Int,
        uniqueFixtureCount: Int,
        reportEntries: List<DirectFixtureCoverageModuleReport>,
    ): Int {
        if (reportEntries.size != directModuleCount) {
            throw GradleException(buildString {
                append("Direct model-family fixture report drift: directModuleCount=$directModuleCount, ")
                append("reportEntryCount=${reportEntries.size}")
            })
        }
        val fixtureCount = reportEntries.sumOf { it.fixtures.size }
        if (fixtureCount != uniqueFixtureCount) {
            throw GradleException(buildString {
                append("Direct model-family fixture report drift: fixtureCount=$fixtureCount, ")
                append("uniqueFixtureCount=$uniqueFixtureCount")
            })
        }
        val incompleteFixtureDetails = reportEntries
            .flatMap { moduleReport ->
                moduleReport.fixtures.filter { fixture ->
                    fixture.id.isBlank()
                            || fixture.modelType.isBlank()
                            || fixture.architectures.isEmpty()
                            || fixture.tokenizerMarker.isBlank()
                }.map { fixture -> "${moduleReport.moduleName}:${fixture.id.ifBlank { "<blank>" }}" }
            }
        if (incompleteFixtureDetails.isNotEmpty()) {
            throw GradleException(buildString {
                append("Direct model-family fixture report drift: incomplete fixtureDetails for ")
                append(incompleteFixtureDetails.joinToString(", "))
            })
        }
        return fixtureCount
    }
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageModuleReport.kt
================================================
package tech.kayys.tafkir.buildlogic

data class DirectFixtureCoverageModuleReport(
    val moduleName: String,
    val fixtures: List<DirectFixtureCoverage>,
)



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageReportContract.kt
================================================
package tech.kayys.tafkir.buildlogic

import groovy.json.JsonSlurper
import org.gradle.api.GradleException
import java.io.File

private typealias Spec = DirectFixtureCoverageReportSpec

object DirectFixtureCoverageReportContract {
    fun requireReportContract(report: Map<*, *>, schema: Map<*, *>): Unit {
        val problems = mutableListOf<String>()
        val schemaProperties = schema["properties"] as? Map<*, *>
        val schemaVersionProperty = schemaProperties?.get("schemaVersion") as? Map<*, *>
        val schemaModulesProperty = schemaProperties?.get("modules") as? Map<*, *>
        val schemaModuleItem = schemaModulesProperty?.get("items") as? Map<*, *>
        val schemaModuleProperties = schemaModuleItem?.get("properties") as? Map<*, *>
        val schemaFixtureDetails = schemaModuleProperties?.get("fixtureDetails") as? Map<*, *>
        val schemaFixtureDetailItem = schemaFixtureDetails?.get("items") as? Map<*, *>
        val schemaFixtureDetailProperties = schemaFixtureDetailItem?.get("properties") as? Map<*, *>
        val schemaTokenizerMarker = schemaFixtureDetailProperties?.get("tokenizerMarker") as? Map<*, *>

        if (schema["\$id"] != Spec.schemaId) {
            problems += "schema \$id should be ${Spec.schemaId}"
        }
        val schemaRootRequired = requiredFieldSet(schema["required"])
        if (schemaRootRequired != Spec.rootFieldSet) {
            problems += fieldDriftMessage("schema root required", Spec.rootFieldSet, schemaRootRequired)
        }
        if ((schemaVersionProperty?.get("const") as? Number)?.toInt() != Spec.schemaVersion) {
            problems += "schemaVersion const should be ${Spec.schemaVersion}"
        }
        val schemaModuleRequired = requiredFieldSet(schemaModuleItem?.get("required"))
        if (schemaModuleRequired != Spec.moduleFieldSet) {
            problems += fieldDriftMessage("schema module required", Spec.moduleFieldSet, schemaModuleRequired)
        }
        val schemaFixtureDetailRequired = requiredFieldSet(schemaFixtureDetailItem?.get("required"))
        if (schemaFixtureDetailRequired != Spec.detailFieldSet) {
            problems += fieldDriftMessage(
                "schema fixtureDetails required",
                Spec.detailFieldSet,
                schemaFixtureDetailRequired,
            )
        }
        if (schemaTokenizerMarker?.get("enum") != Spec.tokenizerMarkers) {
            problems += "schema tokenizerMarker enum drifted"
        }

        val reportFields = fieldSet(report)
        if (reportFields != Spec.rootFieldSet) {
            problems += fieldDriftMessage("report root", Spec.rootFieldSet, reportFields)
        }
        if ((report["schemaVersion"] as? Number)?.toInt() != Spec.schemaVersion) {
            problems += "report schemaVersion should be ${Spec.schemaVersion}"
        }
        val directModuleCount = (report["directModuleCount"] as? Number)?.toInt()
        val fixtureCount = (report["fixtureCount"] as? Number)?.toInt()
        val modules = report["modules"] as? List<*>
        if (directModuleCount == null || directModuleCount < 1) {
            problems += "report directModuleCount should be a positive integer"
        }
        if (fixtureCount == null || fixtureCount < 1) {
            problems += "report fixtureCount should be a positive integer"
        }
        if (modules == null || modules.isEmpty()) {
            problems += "report modules should be a non-empty list"
        } else {
            if (directModuleCount != modules.size) {
                problems += "report directModuleCount=$directModuleCount but modules.size=${modules.size}"
            }
            var observedFixtureCount = 0
            modules.forEach { moduleEntry ->
                val module = moduleEntry as? Map<*, *>
                if (module == null) {
                    problems += "report module entry should be an object"
                    return@forEach
                }
                val moduleFields = fieldSet(module)
                if (moduleFields != Spec.moduleFieldSet) {
                    problems += fieldDriftMessage(
                        "report module ${module["module"]}",
                        Spec.moduleFieldSet,
                        moduleFields,
                    )
                }
                val moduleName = module["module"]?.toString().orEmpty()
                if (!moduleName.matches(Spec.moduleNameRegex)) {
                    problems += "report module '$moduleName' should match tafkir-model-* naming"
                }
                val moduleFixtureCount = (module["fixtureCount"] as? Number)?.toInt()
                val fixtures = (module["fixtures"] as? List<*>)?.map { it?.toString().orEmpty() }
                val fixtureDetails = module["fixtureDetails"] as? List<*>
                if (moduleFixtureCount == null || moduleFixtureCount < 1) {
                    problems += "$moduleName fixtureCount should be a positive integer"
                }
                if (fixtures == null || fixtures.any { it.isBlank() }) {
                    problems += "$moduleName fixtures should be non-blank strings"
                }
                if (fixtureDetails == null || fixtureDetails.isEmpty()) {
                    problems += "$moduleName fixtureDetails should be non-empty"
                }
                if (fixtures != null && moduleFixtureCount != fixtures.size) {
                    problems += "$moduleName fixtureCount=$moduleFixtureCount but fixtures.size=${fixtures.size}"
                }
                if (fixtureDetails != null && moduleFixtureCount != fixtureDetails.size) {
                    problems += "$moduleName fixtureCount=$moduleFixtureCount but fixtureDetails.size=${fixtureDetails.size}"
                }
                val detailIds = fixtureDetails
                    ?.mapNotNull { detail -> (detail as? Map<*, *>)?.get("id")?.toString() }
                    ?: emptyList()
                if (fixtures != null && fixtures != detailIds) {
                    problems += "$moduleName fixtures and fixtureDetails ids differ"
                }
                observedFixtureCount += moduleFixtureCount ?: 0
                fixtureDetails?.forEach { detailEntry ->
                    val detail = detailEntry as? Map<*, *>
                    if (detail == null) {
                        problems += "$moduleName fixtureDetail should be an object"
                        return@forEach
                    }
                    val detailFields = fieldSet(detail)
                    if (detailFields != Spec.detailFieldSet) {
                        problems += fieldDriftMessage(
                            "$moduleName fixtureDetail ${detail["id"]}",
                            Spec.detailFieldSet,
                            detailFields,
                        )
                    }
                    val id = detail["id"]?.toString().orEmpty()
                    val modelType = detail["modelType"]?.toString().orEmpty()
                    val architectures = (detail["architectures"] as? List<*>)?.map { it?.toString().orEmpty() }
                    val tokenizerMarker = detail["tokenizerMarker"]?.toString().orEmpty()
                    if (id.isBlank() || modelType.isBlank()) {
                        problems += "$moduleName fixtureDetail id/modelType should be non-blank"
                    }
                    if (architectures == null || architectures.isEmpty() || architectures.any { it.isBlank() }) {
                        problems += "$moduleName:$id architectures should be non-empty strings"
                    }
                    if (tokenizerMarker !in Spec.tokenizerMarkers) {
                        problems += "$moduleName:$id tokenizerMarker '$tokenizerMarker' is unsupported"
                    }
                }
            }
            if (fixtureCount != observedFixtureCount) {
                problems += "report fixtureCount=$fixtureCount but observedFixtureCount=$observedFixtureCount"
            }
        }

        if (problems.isNotEmpty()) {
            throw GradleException(
                "Direct model-family fixture report contract problems:\n" +
                        problems.joinToString(separator = "\n") { "- $it" },
            )
        }
    }

    fun requireGeneratedReports(
        report: Map<*, *>,
        schema: Map<*, *>,
        reportFile: File,
        schemaFile: File,
    ): Unit {
        requireReportContract(report, schema)
        requireReportContract(
            parseJsonFileMap(reportFile),
            parseJsonFileMap(schemaFile),
        )
    }

    fun expectContractFailure(
        description: String,
        expectedMessage: String,
        report: Map<*, *>,
        schema: Map<*, *>,
    ): Unit {
        val failure = try {
            requireReportContract(report, schema)
            null
        } catch (error: GradleException) {
            error
        }
        if (failure == null) {
            throw GradleException("Expected direct fixture coverage contract failure for $description")
        }
        if (!failure.message.orEmpty().contains(expectedMessage)) {
            throw GradleException(buildString {
                append("Expected direct fixture coverage contract failure for $description to contain '$expectedMessage' ")
                append("but was: ${failure.message}")
            })
        }
    }

    private fun fieldDriftMessage(scope: String, expected: Set<String>, actual: Set<String>?): String =
        "$scope fields drifted; expected=${expected.sorted()}, actual=${actual?.sorted() ?: "<missing>"}"

    private fun fieldSet(value: Map<*, *>): Set<String> =
        value.keys.map { it.toString() }.toSet()

    private fun requiredFieldSet(value: Any?): Set<String>? =
        (value as? List<*>)?.map { it.toString() }?.toSet()

    private fun parseJsonFileMap(file: File): Map<*, *> =
        JsonSlurper().parse(file) as Map<*, *>
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageReportFailureSmoke.kt
================================================
package tech.kayys.tafkir.buildlogic

import org.gradle.api.GradleException

object DirectFixtureCoverageReportFailureSmoke {
    private const val fixtureCountDriftDescription = "fixture count drift"
    private const val fixtureCountDriftMessage = "report fixtureCount=2 but observedFixtureCount=1"
    private const val fixtureIdDetailDriftDescription = "fixture id/detail drift"
    private const val fixtureIdDetailDriftMessage = "fixtures and fixtureDetails ids differ"
    private const val unsupportedTokenizerDescription = "unsupported tokenizer marker"
    private val unsupportedTokenizerMessage =
        "tokenizerMarker '${DirectFixtureCoverageSmokeFixtures.unsupportedTokenizerMarker}' is unsupported"
    private const val incompleteFixtureDetailsMessage = "incomplete fixtureDetails"

    fun requireSmokeContract(reportPayload: Map<String, Any>, schemaPayload: Map<String, Any>): Unit {
        contractFailureCases(reportPayload).forEach { failureCase ->
            expectContractFailure(failureCase, schemaPayload)
        }
        requireIncompleteReportEntryFailure()
    }

    private fun contractFailureCases(reportPayload: Map<String, Any>): List<ContractFailureCase> =
        listOf(
            fixtureCountDriftCase(reportPayload),
            fixtureIdDetailDriftCase(reportPayload),
            unsupportedTokenizerCase(reportPayload),
        )

    private fun fixtureCountDriftCase(reportPayload: Map<String, Any>): ContractFailureCase =
        ContractFailureCase(
            description = fixtureCountDriftDescription,
            expectedMessage = fixtureCountDriftMessage,
            report = reportPayload + ("fixtureCount" to 2),
        )

    private fun fixtureIdDetailDriftCase(reportPayload: Map<String, Any>): ContractFailureCase =
        ContractFailureCase(
            description = fixtureIdDetailDriftDescription,
            expectedMessage = fixtureIdDetailDriftMessage,
            report = reportPayload + ("modules" to listOf(
                DirectFixtureCoverageSmokeFixtures.mismatchedFixtureDetailModuleMap(),
            )),
        )

    private fun unsupportedTokenizerCase(reportPayload: Map<String, Any>): ContractFailureCase =
        ContractFailureCase(
            description = unsupportedTokenizerDescription,
            expectedMessage = unsupportedTokenizerMessage,
            report = reportPayload + ("modules" to listOf(
                DirectFixtureCoverageSmokeFixtures.unsupportedTokenizerModuleMap(),
            )),
        )

    private fun requireIncompleteReportEntryFailure(): Unit {
        val incompleteFailure = try {
            DirectFixtureCoverageGuards.requireReportEntries(
                directModuleCount = 1,
                uniqueFixtureCount = 1,
                reportEntries = DirectFixtureCoverageSmokeFixtures.incompleteReportEntries(),
            )
            null
        } catch (error: GradleException) {
            error
        }
        if (!incompleteFailure?.message.orEmpty().contains(incompleteFixtureDetailsMessage)) {
            throw GradleException("Direct fixture coverage report entry smoke failure drifted")
        }
    }

    private fun expectContractFailure(failureCase: ContractFailureCase, schema: Map<*, *>): Unit =
        DirectFixtureCoverageReportContract.expectContractFailure(
            description = failureCase.description,
            expectedMessage = failureCase.expectedMessage,
            report = failureCase.report,
            schema = schema,
        )

    private data class ContractFailureCase(
        val description: String,
        val expectedMessage: String,
        val report: Map<*, *>,
    )
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageReportRendering.kt
================================================
package tech.kayys.tafkir.buildlogic

import groovy.json.JsonOutput
import groovy.json.JsonSlurper

object DirectFixtureCoverageReportRendering {
    fun prettyJsonWithTrailingNewline(value: Any?): String =
        buildString {
            appendLine(JsonOutput.prettyPrint(JsonOutput.toJson(value)))
        }

    fun jsonRoundTripMap(value: Any?): Map<*, *> =
        JsonSlurper().parseText(prettyJsonWithTrailingNewline(value)) as Map<*, *>

    fun reportPayload(
        directModuleCount: Int,
        fixtureCount: Int,
        reportEntries: List<DirectFixtureCoverageModuleReport>,
    ): Map<String, Any> = mapOf(
        "schemaVersion" to DirectFixtureCoverageReportSpec.schemaVersion,
        "directModuleCount" to directModuleCount,
        "fixtureCount" to fixtureCount,
        "modules" to reportEntries
            .sortedBy { it.moduleName }
            .map { moduleReport ->
                mapOf(
                    "module" to moduleReport.moduleName,
                    "fixtureCount" to moduleReport.fixtures.size,
                    "fixtures" to moduleReport.fixtures.map { it.id },
                    "fixtureDetails" to moduleReport.fixtures.map { fixture ->
                        mapOf(
                            "id" to fixture.id,
                            "modelType" to fixture.modelType,
                            "architectures" to fixture.architectures,
                            "tokenizerMarker" to fixture.tokenizerMarker,
                        )
                    },
                )
            },
    )

    fun textReport(
        directModuleCount: Int,
        fixtureCount: Int,
        reportEntries: List<DirectFixtureCoverageModuleReport>,
    ): String = buildString {
        appendLine("Direct model-family fixture coverage")
        appendLine("schemaVersion=${DirectFixtureCoverageReportSpec.schemaVersion}")
        appendLine("directModuleCount=$directModuleCount")
        appendLine("fixtureCount=$fixtureCount")
        appendLine()
        reportEntries
            .sortedBy { it.moduleName }
            .forEach { moduleReport ->
                appendLine(
                    "${moduleReport.moduleName} (${moduleReport.fixtures.size}): " +
                            moduleReport.fixtures.joinToString(", ") { it.id },
                )
                moduleReport.fixtures.forEach { fixture ->
                    appendLine(buildString {
                        append("  - ${fixture.id}: model_type=${fixture.modelType}, ")
                        append("architectures=${fixture.architectures.joinToString("|")}, ")
                        append("tokenizer=${fixture.tokenizerMarker}")
                    })
                }
            }
    }

    fun schemaPayload(): Map<String, Any> = mapOf(
        "\$schema" to "https://json-schema.org/draft/2020-12/schema",
        "\$id" to DirectFixtureCoverageReportSpec.schemaId,
        "title" to "Tafkir direct model-family fixture coverage report",
        "type" to "object",
        "additionalProperties" to false,
        "required" to DirectFixtureCoverageReportSpec.rootFields,
        "properties" to mapOf(
            "schemaVersion" to mapOf("const" to DirectFixtureCoverageReportSpec.schemaVersion),
            "directModuleCount" to mapOf("type" to "integer", "minimum" to 1),
            "fixtureCount" to mapOf("type" to "integer", "minimum" to 1),
            "modules" to mapOf(
                "type" to "array",
                "minItems" to 1,
                "items" to mapOf(
                    "type" to "object",
                    "additionalProperties" to false,
                    "required" to DirectFixtureCoverageReportSpec.moduleFields,
                    "properties" to mapOf(
                        "module" to mapOf(
                            "type" to "string",
                            "pattern" to DirectFixtureCoverageReportSpec.moduleNamePattern,
                        ),
                        "fixtureCount" to mapOf("type" to "integer", "minimum" to 1),
                        "fixtures" to mapOf(
                            "type" to "array",
                            "minItems" to 1,
                            "items" to mapOf("type" to "string", "minLength" to 1),
                        ),
                        "fixtureDetails" to mapOf(
                            "type" to "array",
                            "minItems" to 1,
                            "items" to mapOf(
                                "type" to "object",
                                "additionalProperties" to false,
                                "required" to DirectFixtureCoverageReportSpec.detailFields,
                                "properties" to mapOf(
                                    "id" to mapOf("type" to "string", "minLength" to 1),
                                    "modelType" to mapOf("type" to "string", "minLength" to 1),
                                    "architectures" to mapOf(
                                        "type" to "array",
                                        "minItems" to 1,
                                        "items" to mapOf("type" to "string", "minLength" to 1),
                                    ),
                                    "tokenizerMarker" to mapOf(
                                        "type" to "string",
                                        "enum" to DirectFixtureCoverageReportSpec.tokenizerMarkers,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageReportSmoke.kt
================================================
package tech.kayys.tafkir.buildlogic

import org.gradle.api.GradleException
import java.nio.file.Files

private typealias SmokeSpec = DirectFixtureCoverageReportSpec
private typealias SmokeFixtures = DirectFixtureCoverageSmokeFixtures

object DirectFixtureCoverageReportSmoke {
    fun requireSmokeContract(): Unit {
        DirectFixtureCoverageSpecSmoke.requireSmokeContract()
        val smokeReportEntries = SmokeFixtures.reportEntries()
        val reportPayload = DirectFixtureCoverageReportRendering.reportPayload(
            directModuleCount = 1,
            fixtureCount = 1,
            reportEntries = smokeReportEntries,
        )
        val schemaPayload = DirectFixtureCoverageReportRendering.schemaPayload()

        requireGuardSmoke(smokeReportEntries)
        requireTextReportSmoke(smokeReportEntries)
        requireReportContractSmoke(reportPayload, schemaPayload)
        requireGeneratedReportSmoke(reportPayload, schemaPayload)
        requireChangeAwareWriterSmoke(smokeReportEntries)
        DirectFixtureCoverageReportFailureSmoke.requireSmokeContract(reportPayload, schemaPayload)
    }

    private fun requireGuardSmoke(smokeReportEntries: List<DirectFixtureCoverageModuleReport>): Unit {
        DirectFixtureCoverageGuards.requireProblemsClear(emptyList())
        val coverageFailure = try {
            DirectFixtureCoverageGuards.requireProblemsClear(
                listOf("${SmokeFixtures.moduleName}: missing fixture"),
            )
            null
        } catch (error: GradleException) {
            error
        }
        if (!coverageFailure?.message.orEmpty().contains("- ${SmokeFixtures.moduleName}: missing fixture")) {
            throw GradleException("Direct fixture coverage problem smoke failure drifted")
        }
        val fixtureCount = DirectFixtureCoverageGuards.requireReportEntries(
            directModuleCount = 1,
            uniqueFixtureCount = 1,
            reportEntries = smokeReportEntries,
        )
        if (fixtureCount != 1) {
            throw GradleException("Direct fixture coverage report entry smoke count drifted")
        }
    }

    private fun requireTextReportSmoke(smokeReportEntries: List<DirectFixtureCoverageModuleReport>): Unit {
        val textReport = DirectFixtureCoverageReportRendering.textReport(
            directModuleCount = 1,
            fixtureCount = 1,
            reportEntries = smokeReportEntries,
        )
        if (!textReport.contains("schemaVersion=${SmokeSpec.schemaVersion}")
            || !textReport.contains("${SmokeFixtures.moduleName} (1): ${SmokeFixtures.fixtureId}")
            || !textReport.contains(
                "model_type=${SmokeFixtures.fixtureModelType}, architectures=${SmokeFixtures.architecture}, " +
                        "tokenizer=${SmokeFixtures.fixtureTokenizerMarker}",
            )
        ) {
            throw GradleException("Direct fixture coverage text report smoke output drifted")
        }
    }

    private fun requireReportContractSmoke(reportPayload: Map<String, Any>, schemaPayload: Map<String, Any>): Unit {
        DirectFixtureCoverageReportContract.requireReportContract(reportPayload, schemaPayload)
        DirectFixtureCoverageReportContract.requireReportContract(
            DirectFixtureCoverageReportRendering.jsonRoundTripMap(reportPayload),
            DirectFixtureCoverageReportRendering.jsonRoundTripMap(schemaPayload),
        )
    }

    private fun requireGeneratedReportSmoke(reportPayload: Map<String, Any>, schemaPayload: Map<String, Any>): Unit {
        val smokeDir = Files.createTempDirectory("tafkir-direct-fixture-coverage-").toFile()
        try {
            val smokeReportFile = smokeDir.resolve("model-family-direct-fixtures.json")
            val smokeSchemaFile = smokeDir.resolve("model-family-direct-fixtures.schema.json")
            smokeReportFile.writeText(DirectFixtureCoverageReportRendering.prettyJsonWithTrailingNewline(reportPayload))
            smokeSchemaFile.writeText(DirectFixtureCoverageReportRendering.prettyJsonWithTrailingNewline(schemaPayload))
            DirectFixtureCoverageReportContract.requireGeneratedReports(
                report = reportPayload,
                schema = schemaPayload,
                reportFile = smokeReportFile,
                schemaFile = smokeSchemaFile,
            )
        } finally {
            smokeDir.deleteRecursively()
        }
    }

    private fun requireChangeAwareWriterSmoke(
        smokeReportEntries: List<DirectFixtureCoverageModuleReport>,
    ): Unit {
        val smokeDir = Files.createTempDirectory("tafkir-direct-fixture-writer-").toFile()
        try {
            val textReportFile = smokeDir.resolve("model-family-direct-fixtures.txt")
            val jsonReportFile = smokeDir.resolve("model-family-direct-fixtures.json")
            val schemaFile = smokeDir.resolve("model-family-direct-fixtures.schema.json")
            val reportFiles = listOf(textReportFile, jsonReportFile, schemaFile)
            DirectFixtureCoverageReportWriter.writeReports(
                directModuleCount = 1,
                fixtureCount = 1,
                reportEntries = smokeReportEntries,
                textReportFile = textReportFile,
                jsonReportFile = jsonReportFile,
                schemaFile = schemaFile,
            )
            val pinnedModifiedAt = 1_700_000_000_000L
            reportFiles.forEach { reportFile ->
                if (!reportFile.setLastModified(pinnedModifiedAt)) {
                    throw GradleException("Direct fixture coverage writer smoke could not pin report timestamp")
                }
            }

            DirectFixtureCoverageReportWriter.writeReports(
                directModuleCount = 1,
                fixtureCount = 1,
                reportEntries = smokeReportEntries,
                textReportFile = textReportFile,
                jsonReportFile = jsonReportFile,
                schemaFile = schemaFile,
            )
            if (reportFiles.any { it.lastModified() != pinnedModifiedAt }) {
                throw GradleException("Direct fixture coverage writer rewrote unchanged reports")
            }
        } finally {
            smokeDir.deleteRecursively()
        }
    }
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageReportSpec.kt
================================================
package tech.kayys.tafkir.buildlogic

object DirectFixtureCoverageReportSpec {
    const val schemaVersion = 2

    val schemaId =
        "https://tafkir.ai/schemas/build/model-family-direct-fixtures.v$schemaVersion.schema.json"

    const val moduleNamePattern = "^tafkir-model-[a-z0-9-]+$"
    const val moduleNamePrefix = "tafkir-model-"
    const val mainJavaSourcePath = "src/main/java"
    const val testJavaSourcePath = "src/test/java"
    const val fixtureRootPath = "src/test/resources/model-family-fixtures"
    const val directInferenceMarker = "DIRECT_SAFETENSOR_INFERENCE"
    const val fixtureValidatorCall = "ModelFamilyFixtureValidator.validate"

    val fixtureInputIncludes = listOf(
        "$moduleNamePrefix*/$mainJavaSourcePath/**/*.java",
        "$moduleNamePrefix*/$testJavaSourcePath/**/*.java",
        "$moduleNamePrefix*/$fixtureRootPath/**",
    )

    val moduleNameRegex = Regex(moduleNamePattern)
    val rootFields = listOf("schemaVersion", "directModuleCount", "fixtureCount", "modules")
    val moduleFields = listOf("module", "fixtureCount", "fixtures", "fixtureDetails")
    val detailFields = listOf("id", "modelType", "architectures", "tokenizerMarker")
    val rootFieldSet = rootFields.toSet()
    val moduleFieldSet = moduleFields.toSet()
    val detailFieldSet = detailFields.toSet()

    val tokenizerMarkers = listOf(
        "tokenizer.json",
        "vocab.json",
        "tokenizer/tokenizer.json",
        "tokenizer/vocab.json",
    )
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageReportWriter.kt
================================================
package tech.kayys.tafkir.buildlogic

import java.io.File

object DirectFixtureCoverageReportWriter {
    fun writeReports(
        directModuleCount: Int,
        fixtureCount: Int,
        reportEntries: List<DirectFixtureCoverageModuleReport>,
        textReportFile: File,
        jsonReportFile: File,
        schemaFile: File,
    ): DirectFixtureCoverageWrittenReports {
        writeTextIfChanged(
            textReportFile,
            DirectFixtureCoverageReportRendering.textReport(
                directModuleCount = directModuleCount,
                fixtureCount = fixtureCount,
                reportEntries = reportEntries,
            ),
        )

        val reportPayload = DirectFixtureCoverageReportRendering.reportPayload(
            directModuleCount = directModuleCount,
            fixtureCount = fixtureCount,
            reportEntries = reportEntries,
        )
        writeTextIfChanged(
            jsonReportFile,
            DirectFixtureCoverageReportRendering.prettyJsonWithTrailingNewline(reportPayload),
        )

        val schemaPayload = DirectFixtureCoverageReportRendering.schemaPayload()
        writeTextIfChanged(
            schemaFile,
            DirectFixtureCoverageReportRendering.prettyJsonWithTrailingNewline(schemaPayload),
        )
        DirectFixtureCoverageReportContract.requireGeneratedReports(reportPayload, schemaPayload, jsonReportFile, schemaFile)

        return DirectFixtureCoverageWrittenReports(
            textReportFile = textReportFile,
            jsonReportFile = jsonReportFile,
            schemaFile = schemaFile,
        )
    }

    private fun writeTextIfChanged(file: File, text: String) {
        if (file.exists() && file.readText() == text) {
            return
        }

        file.parentFile.mkdirs()
        file.writeText(text)
    }
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageScanner.kt
================================================
package tech.kayys.tafkir.buildlogic

import java.io.File

object DirectFixtureCoverageScanner {
    fun scan(modelModulesDir: File): DirectFixtureCoverageScanResult {
        val modules = modelModulesDir.listFiles()
            ?.filter { it.isDirectory && it.name.startsWith(DirectFixtureCoverageReportSpec.moduleNamePrefix) }
            ?.sortedBy { it.name }
            ?: emptyList()

        val problems = mutableListOf<String>()
        val reportEntries = mutableListOf<DirectFixtureCoverageModuleReport>()
        val fixtureOwners = linkedMapOf<String, String>()
        var directModuleCount = 0

        modules.forEach { module ->
            val mainJavaText = DirectFixtureCoverageFileSupport.javaSourceTextIfContains(
                module,
                DirectFixtureCoverageReportSpec.mainJavaSourcePath,
                DirectFixtureCoverageReportSpec.directInferenceMarker,
            )
            if (mainJavaText == null) {
                return@forEach
            }

            directModuleCount += 1
            scanDirectModule(
                module = module,
                mainJavaText = mainJavaText,
                hasFixtureValidatorCall = DirectFixtureCoverageFileSupport.containsJavaSourceText(
                    module,
                    DirectFixtureCoverageReportSpec.testJavaSourcePath,
                    DirectFixtureCoverageReportSpec.fixtureValidatorCall,
                ),
                fixtureOwners = fixtureOwners,
                reportEntries = reportEntries,
                problems = problems,
            )
        }
        if (directModuleCount == 0) {
            problems += "no direct-safetensor model-family modules were detected under $modelModulesDir"
        }

        return DirectFixtureCoverageScanResult(
            modelModuleCount = modules.size,
            directModuleCount = directModuleCount,
            uniqueFixtureCount = fixtureOwners.size,
            reportEntries = reportEntries,
            problems = problems,
        )
    }

    private fun scanDirectModule(
        module: File,
        mainJavaText: String,
        hasFixtureValidatorCall: Boolean,
        fixtureOwners: MutableMap<String, String>,
        reportEntries: MutableList<DirectFixtureCoverageModuleReport>,
        problems: MutableList<String>,
    ): Unit {
        val fixtureRoot = module.resolve(DirectFixtureCoverageReportSpec.fixtureRootPath)
        val fixtureConfigs = fixtureConfigs(fixtureRoot)
        if (fixtureConfigs.isEmpty()) {
            problems += "${module.name}: missing ${DirectFixtureCoverageReportSpec.fixtureRootPath}/*/config.json"
        } else {
            registerFixtureOwners(module, fixtureConfigs, fixtureOwners, problems)
        }

        val moduleFixtureDetails = fixtureConfigs.mapNotNull { config ->
            DirectFixtureCoverageFixtureParser.parse(module, mainJavaText, config, problems)
        }
        if (fixtureConfigs.isNotEmpty()) {
            reportEntries += DirectFixtureCoverageModuleReport(
                moduleName = module.name,
                fixtures = moduleFixtureDetails.sortedBy { it.id },
            )
        }

        if (!hasFixtureValidatorCall) {
            problems += "${module.name}: tests should call ${DirectFixtureCoverageReportSpec.fixtureValidatorCall}"
        }
    }

    private fun registerFixtureOwners(
        module: File,
        fixtureConfigs: List<File>,
        fixtureOwners: MutableMap<String, String>,
        problems: MutableList<String>,
    ): Unit {
        fixtureConfigs.map { it.parentFile.name }.forEach { fixtureName ->
            val previousOwner = fixtureOwners.putIfAbsent(fixtureName, module.name)
            if (previousOwner != null) {
                problems += "fixture '$fixtureName' is declared by both $previousOwner and ${module.name}"
            }
        }
    }

    private fun fixtureConfigs(fixtureRoot: File): List<File> {
        if (!fixtureRoot.exists()) {
            return emptyList()
        }

        return fixtureRoot.listFiles()
            ?.asSequence()
            ?.filter { it.isDirectory }
            ?.map { it.resolve("config.json") }
            ?.filter { it.isFile }
            ?.sortedBy { it.parentFile.name }
            ?.toList()
            ?: emptyList()
    }
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageScanResult.kt
================================================
package tech.kayys.tafkir.buildlogic

data class DirectFixtureCoverageScanResult(
    val modelModuleCount: Int,
    val directModuleCount: Int,
    val uniqueFixtureCount: Int,
    val reportEntries: List<DirectFixtureCoverageModuleReport>,
    val problems: List<String>,
)



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageSmokeFixtures.kt
================================================
package tech.kayys.tafkir.buildlogic

object DirectFixtureCoverageSmokeFixtures {
    const val moduleName = "tafkir-model-smoke"
    const val fixtureId = "smoke"
    const val fixtureModelType = "smoke"
    const val architecture = "SmokeForCausalLM"
    const val fixtureTokenizerMarker = "tokenizer.json"
    const val mismatchedFixtureDetailId = "other"
    const val unsupportedTokenizerMarker = "merges.txt"

    fun reportEntries(): List<DirectFixtureCoverageModuleReport> =
        listOf(
            DirectFixtureCoverageModuleReport(
                moduleName = moduleName,
                fixtures = listOf(fixture()),
            ),
        )

    fun incompleteReportEntries(): List<DirectFixtureCoverageModuleReport> =
        listOf(
            DirectFixtureCoverageModuleReport(
                moduleName = moduleName,
                fixtures = listOf(fixture(modelType = "")),
            ),
        )

    fun fixture(modelType: String = fixtureModelType): DirectFixtureCoverage =
        DirectFixtureCoverage(
            id = fixtureId,
            modelType = modelType,
            architectures = listOf(architecture),
            tokenizerMarker = fixtureTokenizerMarker,
        )

    fun reportModuleMap(
        fixtureDetailId: String = fixtureId,
        tokenizerMarker: String = fixtureTokenizerMarker,
    ): Map<String, Any> =
        mapOf(
            "module" to moduleName,
            "fixtureCount" to 1,
            "fixtures" to listOf(fixtureId),
            "fixtureDetails" to listOf(
                mapOf(
                    "id" to fixtureDetailId,
                    "modelType" to fixtureModelType,
                    "architectures" to listOf(architecture),
                    "tokenizerMarker" to tokenizerMarker,
                ),
            ),
        )

    fun mismatchedFixtureDetailModuleMap(): Map<String, Any> =
        reportModuleMap(fixtureDetailId = mismatchedFixtureDetailId)

    fun unsupportedTokenizerModuleMap(): Map<String, Any> =
        reportModuleMap(tokenizerMarker = unsupportedTokenizerMarker)
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageSpecSmoke.kt
================================================
package tech.kayys.tafkir.buildlogic

import org.gradle.api.GradleException
import java.io.File
import java.nio.file.Files

private typealias SpecSmokeFixtures = DirectFixtureCoverageSmokeFixtures

object DirectFixtureCoverageSpecSmoke {
    fun requireSmokeContract(): Unit {
        val spec = DirectFixtureCoverageReportSpec
        if (!spec.moduleNameRegex.matches("${spec.moduleNamePrefix}smoke")) {
            throw GradleException("Direct fixture coverage module prefix no longer matches module regex")
        }
        if (spec.fixtureInputIncludes != listOf(
                "${spec.moduleNamePrefix}*/${spec.mainJavaSourcePath}/**/*.java",
                "${spec.moduleNamePrefix}*/${spec.testJavaSourcePath}/**/*.java",
                "${spec.moduleNamePrefix}*/${spec.fixtureRootPath}/**",
            )
        ) {
            throw GradleException("Direct fixture coverage input include spec drifted")
        }
        if (spec.directInferenceMarker.isBlank() || spec.fixtureValidatorCall.isBlank()) {
            throw GradleException("Direct fixture coverage scanner marker spec drifted")
        }
        requireFileSupportSmoke(spec)
        requireScannerCountSmoke(spec)
        requireTaskMessageSmoke()
    }

    private fun requireFileSupportSmoke(spec: DirectFixtureCoverageReportSpec): Unit {
        val smokeDir = Files.createTempDirectory("tafkir-direct-fixture-source-").toFile()
        try {
            val directModule = smokeDir.resolve("direct")
            val nonDirectModule = smokeDir.resolve("non-direct")
            writeJavaSource(
                directModule,
                spec.mainJavaSourcePath,
                "z/Direct.java",
                "final class Direct { static final String MODE = \"${spec.directInferenceMarker}\"; }",
            )
            writeJavaSource(
                directModule,
                spec.mainJavaSourcePath,
                "a/Before.java",
                "final class Before {}",
            )
            writeJavaSource(
                nonDirectModule,
                spec.mainJavaSourcePath,
                "Only.java",
                "final class Only {}",
            )

            if (DirectFixtureCoverageFileSupport.containsJavaSourceText(
                    nonDirectModule,
                    spec.mainJavaSourcePath,
                    spec.directInferenceMarker,
                )
            ) {
                throw GradleException("Direct fixture coverage source smoke detected a missing marker")
            }
            val directSource = DirectFixtureCoverageFileSupport.javaSourceTextIfContains(
                directModule,
                spec.mainJavaSourcePath,
                spec.directInferenceMarker,
            ) ?: throw GradleException("Direct fixture coverage source smoke missed a direct marker")
            if (DirectFixtureCoverageFileSupport.javaSourceTextIfContains(
                    nonDirectModule,
                    spec.mainJavaSourcePath,
                    spec.directInferenceMarker,
                ) != null
            ) {
                throw GradleException("Direct fixture coverage source smoke aggregated a non-direct module")
            }
            if (directSource.indexOf("final class Before") !in 0 until directSource.indexOf("final class Direct")) {
                throw GradleException("Direct fixture coverage source smoke ordering drifted")
            }
        } finally {
            smokeDir.deleteRecursively()
        }
    }

    private fun writeJavaSource(module: File, relativeDir: String, relativePath: String, text: String) {
        val sourceFile = module.resolve(relativeDir).resolve(relativePath)
        sourceFile.parentFile.mkdirs()
        sourceFile.writeText(text)
    }

    private fun requireScannerCountSmoke(spec: DirectFixtureCoverageReportSpec): Unit {
        val smokeDir = Files.createTempDirectory("tafkir-direct-fixture-scanner-").toFile()
        try {
            val directModule = smokeDir.resolve("${spec.moduleNamePrefix}direct-smoke")
            val nonDirectModule = smokeDir.resolve("${spec.moduleNamePrefix}plain-smoke")
            writeJavaSource(
                directModule,
                spec.mainJavaSourcePath,
                "DirectSmoke.java",
                """
                final class DirectSmoke {
                    static final String MARKER = "${spec.directInferenceMarker}";
                    static final String MODEL = "${SpecSmokeFixtures.fixtureModelType}";
                    static final String ARCH = "${SpecSmokeFixtures.architecture}";
                }
                """.trimIndent(),
            )
            writeJavaSource(
                directModule,
                spec.testJavaSourcePath,
                "DirectSmokeTest.java",
                "final class DirectSmokeTest { void test() { ${spec.fixtureValidatorCall}(null); } }",
            )
            writeFixtureConfig(directModule, spec)
            writeJavaSource(
                nonDirectModule,
                spec.mainJavaSourcePath,
                "PlainSmoke.java",
                "final class PlainSmoke {}",
            )

            val scan = DirectFixtureCoverageScanner.scan(smokeDir)
            DirectFixtureCoverageGuards.requireProblemsClear(scan.problems)
            if (scan.modelModuleCount != 2 || scan.directModuleCount != 1 || scan.uniqueFixtureCount != 1) {
                throw GradleException(
                    "Direct fixture coverage scanner count smoke drifted: " +
                            "modelModuleCount=${scan.modelModuleCount}, " +
                            "directModuleCount=${scan.directModuleCount}, " +
                            "uniqueFixtureCount=${scan.uniqueFixtureCount}",
                )
            }
            if (scan.reportEntries.map { it.moduleName } != listOf(directModule.name)) {
                throw GradleException(
                    "Direct fixture coverage scanner report module smoke drifted: " +
                            "modules=${scan.reportEntries.map { it.moduleName }}",
                )
            }
            if (scan.reportEntries.single().fixtures.map { it.id } != listOf(SpecSmokeFixtures.fixtureId)) {
                throw GradleException(
                    "Direct fixture coverage scanner fixture id smoke drifted: " +
                            "fixtureIds=${scan.reportEntries.single().fixtures.map { it.id }}",
                )
            }
        } finally {
            smokeDir.deleteRecursively()
        }
    }

    private fun writeFixtureConfig(module: File, spec: DirectFixtureCoverageReportSpec) {
        val fixtureDir = module.resolve(spec.fixtureRootPath).resolve(SpecSmokeFixtures.fixtureId)
        fixtureDir.mkdirs()
        fixtureDir.resolve("config.json").writeText(
            """
            {
              "model_type": "${SpecSmokeFixtures.fixtureModelType}",
              "architectures": ["${SpecSmokeFixtures.architecture}"]
            }
            """.trimIndent(),
        )
        fixtureDir.resolve(SpecSmokeFixtures.fixtureTokenizerMarker).writeText("{}")
    }

    private fun requireTaskMessageSmoke(): Unit {
        val smokeDir = Files.createTempDirectory("tafkir-direct-fixture-messages-").toFile()
        try {
            val writtenReports = DirectFixtureCoverageWrittenReports(
                textReportFile = smokeDir.resolve("report.txt"),
                jsonReportFile = smokeDir.resolve("report.json"),
                schemaFile = smokeDir.resolve("report.schema.json"),
            )
            val scan = DirectFixtureCoverageScanResult(
                modelModuleCount = 2,
                directModuleCount = 1,
                uniqueFixtureCount = 1,
                reportEntries = SpecSmokeFixtures.reportEntries(),
                problems = emptyList(),
            )
            val validationSummary = DirectFixtureCoverageTaskMessages.validationSummary(
                scan = scan,
                writtenReports = writtenReports,
                projectDir = smokeDir,
            )
            if (!validationSummary.contains("1 direct model-family module")
                || !validationSummary.contains("report.txt, report.json, report.schema.json")
            ) {
                throw GradleException(
                    "Direct fixture coverage task validation summary smoke drifted: $validationSummary",
                )
            }
            val timingSummary = DirectFixtureCoverageTaskMessages.timingSummary(
                scan = scan,
                startedAt = 0,
                scannedAt = 1_000_000,
                guardedAt = 3_000_000,
                reportedAt = 6_000_000,
            )
            if (!timingSummary.contains("modules=2, directModules=1, fixtures=1")
                || !timingSummary.contains("scan=1ms, guard=2ms, report=3ms, total=6ms")
            ) {
                throw GradleException(
                    "Direct fixture coverage task timing summary smoke drifted: $timingSummary",
                )
            }
        } finally {
            smokeDir.deleteRecursively()
        }
    }
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageTaskMessages.kt
================================================
package tech.kayys.tafkir.buildlogic

import java.io.File

object DirectFixtureCoverageTaskMessages {
    fun validationSummary(
        scan: DirectFixtureCoverageScanResult,
        writtenReports: DirectFixtureCoverageWrittenReports,
        projectDir: File,
    ): String =
        "Validated fixture coverage for ${scan.directModuleCount} direct model-family " +
                "module${pluralSuffix(scan.directModuleCount)}. " +
                "Reports: ${writtenReports.relativeSummary(projectDir)}"

    fun timingSummary(
        scan: DirectFixtureCoverageScanResult,
        startedAt: Long,
        scannedAt: Long,
        guardedAt: Long,
        reportedAt: Long,
    ): String =
        "Direct fixture coverage timings: modules=${scan.modelModuleCount}, " +
                "directModules=${scan.directModuleCount}, " +
                "fixtures=${scan.uniqueFixtureCount}, " +
                "scan=${elapsedMillis(startedAt, scannedAt)}ms, " +
                "guard=${elapsedMillis(scannedAt, guardedAt)}ms, " +
                "report=${elapsedMillis(guardedAt, reportedAt)}ms, " +
                "total=${elapsedMillis(startedAt, reportedAt)}ms"

    private fun elapsedMillis(startedAt: Long, finishedAt: Long): Long =
        (finishedAt - startedAt) / 1_000_000

    private fun pluralSuffix(count: Int): String =
        if (count == 1) "" else "s"
}



================================================
FILE: buildSrc/src/main/kotlin/tech/kayys/tafkir/buildlogic/DirectFixtureCoverageWrittenReports.kt
================================================
package tech.kayys.tafkir.buildlogic

import java.io.File

data class DirectFixtureCoverageWrittenReports(
    val textReportFile: File,
    val jsonReportFile: File,
    val schemaFile: File,
) {
    fun relativeSummary(rootDir: File): String =
        listOf(textReportFile, jsonReportFile, schemaFile)
            .joinToString(separator = ", ") { it.relativeTo(rootDir).path }
}



================================================
FILE: compiler/tafkir-compiler/build.gradle.kts
================================================
plugins {
    java
}

dependencies {
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-ir:0.1.0-SNAPSHOT")
}



================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/CompiledGraph.java
================================================
package tech.kayys.tafkir.compiler;

import java.util.Map;
import tech.kayys.tafkir.ir.*;

public final class CompiledGraph {
    public final GGraph graph;
    public final Map<GValueId, MemorySlot> memory;
    public final ExecutionPlan plan;

    public CompiledGraph(GGraph graph,
            Map<GValueId, MemorySlot> memory,
            ExecutionPlan plan) {
        this.graph = graph;
        this.memory = memory;
        this.plan = plan;
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/Compiler.java
================================================
package tech.kayys.tafkir.compiler;

import tech.kayys.tafkir.ir.*;

public final class Compiler {
    private final GraphOptimizer optimizer = new GraphOptimizer();
    private final MemoryPlanner memory = new MemoryPlanner();
    private final KernelScheduler scheduler = new KernelScheduler();

    public CompiledGraph compile(GGraph graph) {
        // 1. optimize graph
        GGraph optimized = optimizer.optimize(graph);
        // 2. memory plan
        var memoryPlan = memory.plan(optimized);
        // 3. execution schedule
        var execPlan = scheduler.schedule(optimized);
        return new CompiledGraph(optimized, memoryPlan, execPlan);
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/ExecutionPlan.java
================================================
package tech.kayys.tafkir.compiler;

import java.util.*;

public final class ExecutionPlan {
    public static final class Step {
        public final String opName;
        public final int device;
        public final List<Integer> deps;

        public Step(String opName, int device, List<Integer> deps) {
            this.opName = opName;
            this.device = device;
            this.deps = deps;
        }
    }

    private final List<Step> steps;

    public ExecutionPlan(List<Step> steps) {
        this.steps = steps;

    }

    public List<Step> steps() {
        return steps;
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/GraphOptimizer.java
================================================
package tech.kayys.tafkir.compiler;

import tech.kayys.tafkir.ir.*;
import java.util.*;

public final class GraphOptimizer {
    public GGraph optimize(GGraph graph) {
        List<GOp> ops = new ArrayList<>(graph.ops());
        ops = fuseElementwise(ops);
        ops = fuseLinear(ops);
        ops = fuseAttention(ops);
        return new GGraph(
            ops,
            graph.inputs(),
            graph.outputs()
        );
    }
    private List<GOp> fuseElementwise(List<GOp> ops) {
        // pattern match: add → gelu → mul
        // replace with fused op
        return ops;
    }
    private List<GOp> fuseLinear(List<GOp> ops) {
        return ops;
    }
    private List<GOp> fuseAttention(List<GOp> ops) {
        return ops;
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/KernelScheduler.java
================================================
package tech.kayys.tafkir.compiler;

import tech.kayys.tafkir.ir.*;
import java.util.*;

public final class KernelScheduler {
    public ExecutionPlan schedule(GGraph graph) {
        List<ExecutionPlan.Step> steps = new ArrayList<>();
        for (int i = 0; i < graph.ops().size(); i++) {
            GOp op = graph.ops().get(i);
            steps.add(new ExecutionPlan.Step(
                    op.name(),
                    assignDevice(op),
                    dependencies(graph, op)));
        }
        return new ExecutionPlan(steps);
    }

    private int assignDevice(GOp op) {
        return 0; // extend with distributed logic
    }

    private List<Integer> dependencies(GGraph graph, GOp op) {
        return List.of(); // fill with DAG dependencies
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/MemoryPlanner.java
================================================
package tech.kayys.tafkir.compiler;

import tech.kayys.tafkir.ir.*;
import java.util.*;
public final class MemoryPlanner {
public Map<GValueId, MemorySlot> plan(GGraph graph) {
Map<GValueId, Integer> lastUse = computeLastUse(graph);
Map<GValueId, MemorySlot> allocation = new HashMap<>();
List<MemorySlot> freeSlots = new ArrayList<>();
int nextSlot = 0;
for (int i = 0; i < graph.ops().size(); i++) {
GOp op = graph.ops().get(i);
for (GValueId out : op.outputs()) {
MemorySlot slot;
if (!freeSlots.isEmpty()) {
slot = freeSlots.remove(0);
} else {
slot = new MemorySlot(nextSlot++);
}
allocation.put(out, slot);
}
for (GValueRef in : op.inputs()) {
if (lastUse.get(in.id()) == i) {
freeSlots.add(allocation.get(in.id()));
}
}
}
return allocation;
}
private Map<GValueId, Integer> computeLastUse(GGraph graph) {
Map<GValueId, Integer> map = new HashMap<>();
List<GOp> ops = graph.ops();
for (int i = 0; i < ops.size(); i++) {
for (GValueRef in : ops.get(i).inputs()) {
map.put(in.id(), i);
}
}
return map;
}
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/MemorySlot.java
================================================
package tech.kayys.tafkir.compiler;

public final class MemorySlot {
private final int id;
public MemorySlot(int id) {
this.id = id;
}
public int id() { return id; }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/Pass.java
================================================
package tech.kayys.tafkir.compiler;

import tech.kayys.tafkir.ir.GGraph;

/**
 * COMPILER GOAL
 * Transform:
 * High-level IR (attention, ffn, etc.)
 * Optimized IR (fused ops, layout-aware)
 * ↓
 * ↓
 * Executable graph
 */
public interface Pass {
    String name();

    GGraph apply(GGraph graph, PassContext ctx);
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/PassContext.java
================================================
package tech.kayys.tafkir.compiler;

import tech.kayys.tafkir.ir.OpSchemaRegistry;

public final class PassContext {
    public final OpSchemaRegistry schemaRegistry;

    public PassContext(OpSchemaRegistry schemaRegistry) {
        this.schemaRegistry = schemaRegistry;
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/passes/AttentionLoweringPass.java
================================================
package tech.kayys.tafkir.compiler.passes;

import tech.kayys.tafkir.compiler.*;
import tech.kayys.tafkir.ir.*;
import java.util.*;

public final class AttentionLoweringPass implements Pass {
    @Override
    public String name() {
        return "attention_lowering";
    }

    @Override
    public GGraph apply(GGraph graph, PassContext ctx) {
        List<GOp> newOps = new ArrayList<>();
        for (GOp op : graph.ops()) {
            if (!op.opType().equals("attention")) {
                newOps.add(op);
                continue;
            }
            GValueRef q = op.inputs().get(0);
            GValueRef k = op.inputs().get(1);
            GValueRef v = op.inputs().get(2);
            GValueId qk = new GValueId(op.name() + "_qk");
            GValueId sm = new GValueId(op.name() + "_sm");
            newOps.add(new GOp(
                    new OpDescriptor(new OpId("matmul")),
                    op.name() + "_qk",
                    List.of(q, k),
                    List.of(qk),
                    Map.of()));
            newOps.add(new GOp(
                    new OpDescriptor(new OpId("softmax")),
                    op.name() + "_softmax",
                    List.of(new GValueRef(qk)),
                    List.of(sm),
                    Map.of()));
            newOps.add(new GOp(
                    new OpDescriptor(new OpId("matmul")),
                    op.name() + "_out",
                    List.of(new GValueRef(sm), v),
                    op.outputs(),
                    Map.of()));
        }
        return new GGraph(newOps, graph.inputs(), graph.outputs());
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/passes/FlashAttentionPass.java
================================================
package tech.kayys.tafkir.compiler.passes;

import tech.kayys.tafkir.compiler.*;
import tech.kayys.tafkir.ir.*;
import java.util.*;

public final class FlashAttentionPass implements Pass {
    @Override
    public String name() {
        return "flash_attention";
    }

    @Override
    public GGraph apply(GGraph graph, PassContext ctx) {
        List<GOp> ops = new ArrayList<>();
        for (GOp op : graph.ops()) {
            if (op.opType().equals("attention")) {
                ops.add(new GOp(
                        new OpDescriptor(new OpId("flash_attention_v3")),
                        op.name(),
                        op.inputs(),
                        op.outputs(),
                        op.attrs()));
            } else {
                ops.add(op);
            }
        }
        return new GGraph(ops, graph.inputs(), graph.outputs());
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/rewrite/RewritePass.java
================================================
package tech.kayys.tafkir.compiler.rewrite;

import tech.kayys.tafkir.compiler.*;
import tech.kayys.tafkir.ir.*;
import java.util.List;

public final class RewritePass implements Pass {
    private final String name;
    private final List<RewriteRule> rules;

    public RewritePass(String name, List<RewriteRule> rules) {
        this.name = name;
        this.rules = rules;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public GGraph apply(GGraph graph, PassContext ctx) {
        boolean changed;
        do {
            changed = false;
            for (int i = 0; i < graph.ops().size(); i++) {
                for (RewriteRule rule : rules) {
                    if (rule.matches(graph, i)) {
                        RewriteResult res = rule.apply(graph, i);
                        graph = res.graph;
                        changed |= res.changed;
                    }
                }
            }
        } while (changed);
        return graph;
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/rewrite/RewriteResult.java
================================================
package tech.kayys.tafkir.compiler.rewrite;

import tech.kayys.tafkir.ir.GGraph;

public final class RewriteResult {
    public final GGraph graph;
    public final boolean changed;

    public RewriteResult(GGraph graph, boolean changed) {
        this.graph = graph;
        this.changed = changed;
    }
}



================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/rewrite/RewriteRule.java
================================================
package tech.kayys.tafkir.compiler.rewrite;

import tech.kayys.tafkir.ir.*;

public interface RewriteRule {
    boolean matches(GGraph graph, int index);

    RewriteResult apply(GGraph graph, int index);
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/compiler/rules/MatMulAddFusionRule.java
================================================
package tech.kayys.tafkir.compiler.rules;

import tech.kayys.tafkir.compiler.rewrite.*;
import tech.kayys.tafkir.ir.*;
import java.util.*;

public final class MatMulAddFusionRule implements RewriteRule {
    @Override
    public boolean matches(GGraph graph, int i) {
        if (i + 1 >= graph.ops().size())
            return false;
        GOp a = graph.ops().get(i);
        GOp b = graph.ops().get(i + 1);
        return a.opType().equals("matmul")
                && b.opType().equals("add")
                && b.inputs().get(0).id().equals(a.outputs().get(0));
    }

    @Override
    public RewriteResult apply(GGraph graph, int i) {
        List<GOp> ops = new ArrayList<>(graph.ops());
        GOp matmul = ops.get(i);
        GOp add = ops.get(i + 1);
        GOp fused = new GOp(
                new OpDescriptor(new OpId("matmul_bias")),
                matmul.name() + "_fused",
                matmul.inputs(),
                add.outputs(),
                new HashMap<>() // Use HashMap to avoid Map.of issues if any, or just Map.of()
        );
        ops.remove(i + 1);
        ops.set(i, fused);
        return new RewriteResult(
                new GGraph(ops, graph.inputs(), graph.outputs()),
                true
        );
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/conversion/OnnxToIR.java
================================================
package tech.kayys.tafkir.conversion;

import tech.kayys.tafkir.ir.*;
import tech.kayys.tafkir.core.tensor.ModelWeightLoader;
import tech.kayys.tafkir.core.tensor.WeightAdapter;
import java.nio.file.Path;

public final class OnnxToIR implements ModelWeightLoader {
    @Override
    public boolean supports(Path path) {
        return path.toString().endsWith(".onnx");
    }

    @Override
    public WeightAdapter load(Path path) {
        throw new UnsupportedOperationException("OnnxToIR not implemented");
    }
}


================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/conversion/SafeTensorToIR.java
================================================
package tech.kayys.tafkir.conversion;

import tech.kayys.tafkir.ir.*;
import tech.kayys.tafkir.core.tensor.*;
import tech.kayys.tafkir.core.tensor.ModelWeightLoader;
import tech.kayys.tafkir.core.tensor.WeightAdapter;
import java.nio.file.Path;
import java.util.*;

public final class SafeTensorToIR implements ModelWeightLoader {
    @Override
    public boolean supports(Path path) {
        return path.toString().endsWith(".safetensors");
    }

    @Override
    public WeightAdapter load(Path path) {
        // TODO: Implement actual loading
        throw new UnsupportedOperationException("SafeTensorToIR not implemented yet");
    }
}



================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/conversion/ggx/GGXBinding.java
================================================
package tech.kayys.tafkir.conversion.ggx;

/**
 * Defines the contract for an input/output within a GGX file.
 * This aligns the serialization format with the runtime GData execution context.
 */
public record GGXBinding(
    String name,
    String type,    // e.g. "tensor", "scalar", "handle"
    String shape
) {}



================================================
FILE: compiler/tafkir-compiler/src/main/java/tech/kayys/tafkir/conversion/ggx/GGXFile.java
================================================
package tech.kayys.tafkir.conversion.ggx;

import java.util.List;

/**
 * Represents the layout of a compiled graph (GGX).
 * Extended to include strictly typed bindings for the new runtime ExecutionContext.
 */
public final class GGXFile {
    public List<GGXBinding> inputs;
    public List<GGXBinding> outputs;
    
    // Existing fields for weights, graph metadata, etc.
}



================================================
FILE: compiler/tafkir-neural-compiler/build.gradle.kts
================================================
plugins {
    java
}

dependencies {
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}



================================================
FILE: compiler/tafkir-neural-compiler/src/main/java/tech/kayys/tafkir/compiler/neural/DatasetCollector.java
================================================
package tech.kayys.tafkir.compiler.neural;

import java.util.ArrayList;
import java.util.List;

public final class DatasetCollector {
    
    public record KernelTrainingSample(
        NeuralCompilerV2.KernelFeatures features,
        NeuralCompilerV2.KernelConfig config,
        double latency
    ) {}

    private final List<KernelTrainingSample> samples = new ArrayList<>();

    public void record(NeuralCompilerV2.KernelFeatures f, NeuralCompilerV2.KernelConfig c, double latency) {
        samples.add(new KernelTrainingSample(f, c, latency));
    }

    public List<KernelTrainingSample> dataset() {
        return List.copyOf(samples);
    }
    
    public void clear() {
        samples.clear();
    }
}



================================================
FILE: compiler/tafkir-neural-compiler/src/main/java/tech/kayys/tafkir/compiler/neural/KernelModel.java
================================================
package tech.kayys.tafkir.compiler.neural;

import java.util.List;

public final class KernelModel {
    private double[] weights = new double[4]; // [numel, tile, vec, bias]

    public double predict(NeuralCompilerV2.KernelFeatures f, NeuralCompilerV2.KernelConfig c) {
        double numel = 1;
        for (long d : f.shape()) numel *= d;
        
        // Featurize: normalize inputs for the model
        double x1 = Math.log1p(numel) / 20.0;
        double x2 = c.tileSize() / 1024.0;
        double x3 = c.vectorWidth() / 16.0;
        
        return weights[0] * x1 + weights[1] * x2 + weights[2] * x3 + weights[3];
    }

    public void train(List<DatasetCollector.KernelTrainingSample> data) {
        // Simple Gradient Descent to update weights based on latency data
        for (int epoch = 0; epoch < 10; epoch++) {
            for (var sample : data) {
                double pred = predict(sample.features(), sample.config());
                double error = pred - sample.latency();
                // Update weights logic...
            }
        }
    }
}



================================================
FILE: compiler/tafkir-neural-compiler/src/main/java/tech/kayys/tafkir/compiler/neural/NeuralCompilerV2.java
================================================
package tech.kayys.tafkir.compiler.neural;

import java.util.Comparator;
import java.util.List;

public final class NeuralCompilerV2 {
    public record KernelFeatures(String opId, long[] shape, String backend) {}
    public record KernelConfig(int tileSize, int vectorWidth) {}

    private final KernelModel model;

    public NeuralCompilerV2(KernelModel model) {
        this.model = model;
    }

    public KernelConfig select(KernelFeatures f, List<KernelConfig> configs) {
        return configs.stream()
            .min(Comparator.comparingDouble(cfg -> model.predict(f, cfg)))
            .orElseThrow();
    }
}



================================================
FILE: core/tafkir-data/build.gradle.kts
================================================
plugins {
    java
    `maven-publish`
}

group = "tech.kayys.tafkir"
version = "0.3.0-SNAPSHOT"

dependencies {
    implementation(project(":ml:tafkir-ml-aljabr"))
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            groupId = "tech.kayys.tafkir"
            artifactId = "tafkir-data"
            version = "0.3.0-SNAPSHOT"
        }
    }
}



================================================
FILE: core/tafkir-data/src/main/java/tech/kayys/tafkir/data/ImageAugmentation.java
================================================
package tech.kayys.tafkir.data;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.Random;

/**
 * Simple data augmentation for image data.
 */
public final class ImageAugmentation {

    private final Random rng;

    public ImageAugmentation(long seed) { this.rng = new Random(seed); }
    public ImageAugmentation() { this.rng = new Random(); }

    public TafkirTensor randomHorizontalFlip(TafkirTensor images, float p) {
        if (rng.nextFloat() >= p) return images;
        float[] data = images.data();
        int batchSize = (int) images.shape()[0];
        float[] flipped = new float[data.length];
        for (int b = 0; b < batchSize; b++) {
            for (int r = 0; r < 28; r++) {
                for (int c = 0; c < 28; c++) {
                    int srcIdx = b * 784 + r * 28 + c;
                    int dstIdx = b * 784 + r * 28 + (27 - c);
                    flipped[dstIdx] = data[srcIdx];
                }
            }
        }
        return TafkirTensor.of(flipped, batchSize, 784);
    }

    public TafkirTensor addNoise(TafkirTensor images, float std) {
        float[] data = images.data();
        float[] noisy = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            noisy[i] = data[i] + (float) (rng.nextGaussian() * std);
            noisy[i] = Math.max(0, Math.min(1, noisy[i]));
        }
        return TafkirTensor.of(noisy, images.shapeArray());
    }
}



================================================
FILE: core/tafkir-data/src/main/java/tech/kayys/tafkir/data/MNISTDataset.java
================================================
package tech.kayys.tafkir.data;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.io.*;
import java.nio.file.*;
import java.util.zip.GZIPInputStream;

/**
 * MNIST dataset loader.
 * Downloads and parses the classic MNIST dataset from Yann LeCun's website.
 */
public final class MNISTDataset {

    private static final String BASE_URL = "http://yann.lecun.com/exdb/mnist/";

    private final TafkirTensor images;  // [N, 784] — flattened 28x28, normalized to [0,1]
    private final TafkirTensor labels;  // [N] — class indices 0-9
    private final int numSamples;

    private MNISTDataset(TafkirTensor images, TafkirTensor labels) {
        this.images = images;
        this.labels = labels;
        this.numSamples = (int) images.shape()[0];
    }

    public TafkirTensor images() { return images; }
    public TafkirTensor labels() { return labels; }
    public int numSamples() { return numSamples; }

    /**
     * Returns a batch of images and labels.
     */
    public TafkirTensor[] batch(int start, int end) {
        if (end > numSamples) end = numSamples;
        int batchSize = end - start;

        float[] imgBatch = new float[batchSize * 784];
        float[] lblBatch = new float[batchSize];

        float[] allImages = images.data();
        float[] allLabels = labels.data();

        System.arraycopy(allImages, start * 784, imgBatch, 0, batchSize * 784);
        System.arraycopy(allLabels, start, lblBatch, 0, batchSize);

        return new TafkirTensor[]{
            TafkirTensor.of(imgBatch, batchSize, 784),
            TafkirTensor.of(lblBatch, batchSize)
        };
    }

    public static MNISTDataset loadTrain(String dataDir) throws IOException {
        return load(dataDir, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz");
    }

    public static MNISTDataset loadTest(String dataDir) throws IOException {
        return load(dataDir, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz");
    }

    private static MNISTDataset load(String dataDir, String imagesFile, String labelsFile) throws IOException {
        Path dir = Paths.get(dataDir);
        Files.createDirectories(dir);

        Path imagesPath = dir.resolve(imagesFile);
        Path labelsPath = dir.resolve(labelsFile);

        if (!Files.exists(imagesPath)) download(BASE_URL + imagesFile, imagesPath);
        if (!Files.exists(labelsPath)) download(BASE_URL + labelsFile, labelsPath);

        float[] images = parseImages(imagesPath);
        float[] labels = parseLabels(labelsPath);
        int numSamples = labels.length;

        for (int i = 0; i < images.length; i++) images[i] = images[i] / 255.0f;

        return new MNISTDataset(
            TafkirTensor.of(images, numSamples, 784),
            TafkirTensor.of(labels, numSamples)
        );
    }

    private static void download(String url, Path dest) throws IOException {
        System.out.println("Downloading " + url + " ...");
        try (InputStream in = new java.net.URL(url).openStream();
             OutputStream out = Files.newOutputStream(dest)) {
            in.transferTo(out);
        }
        System.out.println("Saved to " + dest);
    }

    private static float[] parseImages(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new GZIPInputStream(Files.newInputStream(path)))) {
            int magic = in.readInt();
            if (magic != 0x00000803) throw new IOException("Invalid MNIST image magic: " + magic);
            int numImages = in.readInt();
            int numRows = in.readInt();
            int numCols = in.readInt();
            if (numRows != 28 || numCols != 28) throw new IOException("Expected 28x28 images");
            float[] data = new float[numImages * 784];
            for (int i = 0; i < data.length; i++) data[i] = in.readUnsignedByte();
            return data;
        }
    }

    private static float[] parseLabels(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new GZIPInputStream(Files.newInputStream(path)))) {
            int magic = in.readInt();
            if (magic != 0x00000801) throw new IOException("Invalid MNIST label magic: " + magic);
            int numLabels = in.readInt();
            float[] data = new float[numLabels];
            for (int i = 0; i < numLabels; i++) data[i] = in.readUnsignedByte();
            return data;
        }
    }
}



================================================
FILE: core/tafkir-distributed/build.gradle.kts
================================================
plugins {
    java
}

dependencies {
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-ir:0.1.0-SNAPSHOT")
    implementation(project(":runtime:tafkir-runtime"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}



================================================
FILE: core/tafkir-distributed/src/main/java/tech/kayys/tafkir/distributed/CommBackend.java
================================================
package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;

public interface CommBackend {
    Tensor allReduce(Tensor t);

    Tensor broadcast(Tensor t, int src);

    Tensor allGather(Tensor t);

    void barrier();
}


================================================
FILE: core/tafkir-distributed/src/main/java/tech/kayys/tafkir/distributed/DDPWrapper.java
================================================
package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;
import java.util.Map;

// DDP (DATA PARALLEL)
public final class DDPWrapper {
    private final CommBackend comm;
    private final DistributedContext ctx;

    public DDPWrapper(CommBackend comm, DistributedContext ctx) {
        this.comm = comm;
        this.ctx = ctx;
    }

    public Map<String, Tensor> syncGradients(Map<String, Tensor> grads) {
        for (String k : grads.keySet()) {
            Tensor g = grads.get(k);
            // sum across all nodes
            Tensor reduced = comm.allReduce(g);
            // average
            grads.put(k, reduced.div(ctx.worldSize()));
        }
        return grads;
    }
}


================================================
FILE: core/tafkir-distributed/src/main/java/tech/kayys/tafkir/distributed/DistributedContext.java
================================================
package tech.kayys.tafkir.distributed;

public final class DistributedContext {
    private final int worldSize;
    private final int rank;

    public DistributedContext(int worldSize, int rank) {
        this.worldSize = worldSize;
        this.rank = rank;
    }

    public int worldSize() {
        return worldSize;
    }

    public int rank() {
        return rank;
    }

    public boolean isMaster() {
        return rank == 0;
    }
}


================================================
FILE: core/tafkir-distributed/src/main/java/tech/kayys/tafkir/distributed/DistributedTrainer.java
================================================
package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;
import tech.kayys.tafkir.ir.*;
import java.util.Map;

/**
 * Distributed trainer - requires tafkir-train module at runtime.
 * Stubbed for compilation.
 */
public final class DistributedTrainer {
    private final Object base; // Trainer - from training module
    private final DDPWrapper ddp;

    public DistributedTrainer(Object base, DDPWrapper ddp) {
        this.base = base;
        this.ddp = ddp;
    }

    public void trainStep(GGraph model,
            GValueId loss,
            Map<String, Tensor> params,
            Map<String, Tensor> inputs) {
        throw new UnsupportedOperationException(
                "DistributedTrainer requires tafkir-train module");
    }
}


================================================
FILE: core/tafkir-distributed/src/main/java/tech/kayys/tafkir/distributed/PipelineStage.java
================================================
package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;

public interface PipelineStage {
    Tensor forward(Tensor input);

    Tensor backward(Tensor grad);
}


================================================
FILE: core/tafkir-distributed/src/main/java/tech/kayys/tafkir/distributed/ShardingPlanner.java
================================================
package tech.kayys.tafkir.distributed;

public final class ShardingPlanner {
    public enum Strategy {
        DDP,
        ZERO,
        TENSOR_PARALLEL,
        PIPELINE
    }

    private final Strategy strategy;

    public ShardingPlanner(Strategy strategy) {
        this.strategy = strategy;
    }

    public Strategy strategy() {
        return strategy;
    }

    public int assignParamShard(String key, int worldSize) {
        return Math.abs(key.hashCode()) % worldSize;
    }
}


================================================
FILE: core/tafkir-distributed/src/main/java/tech/kayys/tafkir/distributed/TensorParallelLinear.java
================================================
package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;

public final class TensorParallelLinear {
    private final CommBackend comm;

    public TensorParallelLinear(CommBackend comm) {
        this.comm = comm;
    }

    public Tensor forward(Tensor x, Tensor localWeight) {
        // partial output
        Tensor out = x.matmul(localWeight);
        // gather all parts
        return comm.allGather(out);
    }
}


================================================
FILE: core/tafkir-distributed/src/main/java/tech/kayys/tafkir/distributed/ZeROOptimizer.java
================================================
package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.core.tensor.Tensor;
import java.util.Map;

public final class ZeROOptimizer {
    private final CommBackend comm;
    private final DistributedContext ctx;

    public ZeROOptimizer(CommBackend comm, DistributedContext ctx) {
        this.comm = comm;
        this.ctx = ctx;
    }

    public void step(Map<String, Tensor> params, Map<String, Tensor> grads) {
        for (String k : params.keySet()) {
            if (!ownsParam(k))
                continue;
            Tensor g = grads.get(k);
            // gather full grad
            Tensor fullGrad = comm.allReduce(g);
            // update shard only
            Tensor p = params.get(k);
            p = p.sub(fullGrad.mul(0.001f)); // simplified
            params.put(k, p);
        }
    }

    private boolean ownsParam(String key) {
        return Math.abs(key.hashCode()) % ctx.worldSize() == ctx.rank();
    }
}


================================================
FILE: core/tafkir-error-code/build.gradle.kts
================================================
plugins {
    java
}

dependencies {
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
}



================================================
FILE: core/tafkir-error-code/src/main/java/tech/kayys/tafkir/error/ErrorCode.java
================================================
package tech.kayys.tafkir.error;

import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Central registry for all Tafkir error codes.
 *
 * <p>
 * Pattern: CATEGORY_NNN (example: MODEL_001)
 * <ul>
 * <li>MODEL: Model-related errors</li>
 * <li>TENSOR: Tensor operation errors</li>
 * <li>DEVICE: Hardware/device errors</li>
 * <li>QUOTA: Resource quota errors</li>
 * <li>AUTH: Authentication/authorization errors</li>
 * <li>INIT: Initialization errors</li>
 * <li>RUNTIME: Runtime execution errors</li>
 * <li>STORAGE: Storage/persistence errors</li>
 * <li>CONVERSION: Model conversion errors</li>
 * <li>VALIDATION: Request validation errors</li>
 * <li>CIRCUIT: Circuit breaker and resilience errors</li>
 * <li>PROVIDER: Provider integration errors</li>
 * <li>ROUTING: Routing and registry errors</li>
 * <li>PLUGIN: Plugin lifecycle errors</li>
 * <li>CONFIG: Configuration errors</li>
 * <li>NETWORK: Network and transport errors</li>
 * <li>STREAM: Streaming errors</li>
 * <li>INTERNAL: Internal/server errors</li>
 * </ul>
 *
 * <p>
 * Rules:
 * <ul>
 * <li>Codes are unique.</li>
 * <li>Code prefix must match its category.</li>
 * </ul>
 *
 * @author Bhangun
 * @since 1.0.0
 */
public enum ErrorCode {

    // ===== Model Errors (404, 400) =====
    MODEL_NOT_FOUND(ErrorCategory.MODEL, 404, "MODEL_001", "Model not found", false),
    MODEL_VERSION_NOT_FOUND(ErrorCategory.MODEL, 404, "MODEL_002", "Model version not found", false),
    MODEL_INVALID_FORMAT(ErrorCategory.MODEL, 400, "MODEL_003", "Invalid model format", false),
    MODEL_CORRUPTED(ErrorCategory.MODEL, 400, "MODEL_004", "Model file corrupted", false),
    MODEL_TOO_LARGE(ErrorCategory.MODEL, 400, "MODEL_005", "Model exceeds size limit", false),
    MODEL_SIGNATURE_INVALID(ErrorCategory.MODEL, 403, "MODEL_006", "Model signature verification failed", false),
    MODEL_NOT_COMPATIBLE(ErrorCategory.MODEL, 400, "MODEL_007", "Model not compatible with selected runner", false),

    // ===== Tensor Errors (400) =====
    TENSOR_SHAPE_MISMATCH(ErrorCategory.TENSOR, 400, "TENSOR_001", "Tensor shape mismatch", false),
    TENSOR_TYPE_MISMATCH(ErrorCategory.TENSOR, 400, "TENSOR_002", "Tensor data type mismatch", false),
    TENSOR_INVALID_DATA(ErrorCategory.TENSOR, 400, "TENSOR_003", "Invalid tensor data", false),
    TENSOR_SIZE_MISMATCH(ErrorCategory.TENSOR, 400, "TENSOR_004", "Tensor size does not match shape", false),
    TENSOR_CONVERSION_FAILED(ErrorCategory.TENSOR, 500, "TENSOR_005", "Tensor conversion failed", true),
    TENSOR_MISSING_INPUT(ErrorCategory.TENSOR, 400, "TENSOR_006", "Required input tensor missing", false),

    // ===== Device Errors (503, 500) =====
    DEVICE_NOT_AVAILABLE(ErrorCategory.DEVICE, 503, "DEVICE_001", "Requested device not available", true),
    DEVICE_OUT_OF_MEMORY(ErrorCategory.DEVICE, 503, "DEVICE_002", "Device out of memory", true),
    DEVICE_INITIALIZATION_FAILED(ErrorCategory.DEVICE, 500, "DEVICE_003", "Device initialization failed", false),
    DEVICE_DRIVER_ERROR(ErrorCategory.DEVICE, 500, "DEVICE_004", "Device driver error", true),
    GPU_NOT_FOUND(ErrorCategory.DEVICE, 503, "DEVICE_005", "GPU not found", false),
    TPU_NOT_AVAILABLE(ErrorCategory.DEVICE, 503, "DEVICE_006", "TPU not available", false),
    NPU_NOT_SUPPORTED(ErrorCategory.DEVICE, 501, "DEVICE_007", "NPU not supported on this platform", false),

    // ===== Quota Errors (429) =====
    QUOTA_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_001", "Quota exceeded", false),
    RATE_LIMIT_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_002", "Rate limit exceeded", false),
    CONCURRENT_REQUESTS_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_003", "Too many concurrent requests", true),
    STORAGE_QUOTA_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_004", "Storage quota exceeded", false),
    COMPUTE_QUOTA_EXCEEDED(ErrorCategory.QUOTA, 429, "QUOTA_005", "Compute quota exceeded", true),

    // ===== Authentication & Authorization (401, 403) =====
    AUTH_TOKEN_INVALID(ErrorCategory.AUTH, 401, "AUTH_001", "Invalid authentication token", false),
    AUTH_TOKEN_EXPIRED(ErrorCategory.AUTH, 401, "AUTH_002", "Authentication token expired", false),
    AUTH_TENANT_NOT_FOUND(ErrorCategory.AUTH, 401, "AUTH_003", "Tenant not found", false),
    AUTH_PERMISSION_DENIED(ErrorCategory.AUTH, 403, "AUTH_004", "Permission denied", false),
    AUTH_TENANT_SUSPENDED(ErrorCategory.AUTH, 403, "AUTH_005", "Tenant account suspended", false),

    // ===== Initialization Errors (500) =====
    INIT_RUNNER_FAILED(ErrorCategory.INIT, 500, "INIT_001", "Runner initialization failed", false),
    INIT_MODEL_LOAD_FAILED(ErrorCategory.INIT, 500, "INIT_002", "Model loading failed", true),
    INIT_NATIVE_LIBRARY_FAILED(ErrorCategory.INIT, 500, "INIT_003", "Native library loading failed", false),
    INIT_CONFIGURATION_INVALID(ErrorCategory.INIT, 500, "INIT_004", "Invalid configuration", false),
    INIT_DEPENDENCY_MISSING(ErrorCategory.INIT, 500, "INIT_005", "Required dependency missing", false),

    // ===== Runtime Execution Errors (500, 504) =====
    RUNTIME_INFERENCE_FAILED(ErrorCategory.RUNTIME, 500, "RUNTIME_001", "Inference execution failed", true),
    RUNTIME_TIMEOUT(ErrorCategory.RUNTIME, 504, "RUNTIME_002", "Inference request timeout", true),
    RUNTIME_OUT_OF_MEMORY(ErrorCategory.RUNTIME, 500, "RUNTIME_003", "Out of memory during inference", true),
    RUNTIME_NATIVE_CRASH(ErrorCategory.RUNTIME, 500, "RUNTIME_004", "Native library crashed", true),
    RUNTIME_INVALID_STATE(ErrorCategory.RUNTIME, 500, "RUNTIME_005", "Invalid runner state", false),
    RUNTIME_BATCH_SIZE_EXCEEDED(ErrorCategory.RUNTIME, 400, "RUNTIME_006", "Batch size exceeds limit", false),

    // ===== Storage Errors (500, 503) =====
    STORAGE_READ_FAILED(ErrorCategory.STORAGE, 500, "STORAGE_001", "Failed to read from storage", true),
    STORAGE_WRITE_FAILED(ErrorCategory.STORAGE, 500, "STORAGE_002", "Failed to write to storage", true),
    STORAGE_NOT_FOUND(ErrorCategory.STORAGE, 404, "STORAGE_003", "Storage resource not found", false),
    STORAGE_CONNECTION_FAILED(ErrorCategory.STORAGE, 503, "STORAGE_004", "Storage connection failed", true),
    STORAGE_PERMISSION_DENIED(ErrorCategory.STORAGE, 403, "STORAGE_005", "Storage permission denied", false),

    // ===== Model Conversion Errors (500) =====
    CONVERSION_FAILED(ErrorCategory.CONVERSION, 500, "CONVERSION_001", "Model conversion failed", true),
    CONVERSION_FORMAT_NOT_SUPPORTED(ErrorCategory.CONVERSION, 400, "CONVERSION_002", "Target format not supported",
            false),
    CONVERSION_TIMEOUT(ErrorCategory.CONVERSION, 504, "CONVERSION_003", "Model conversion timeout", true),
    CONVERSION_VALIDATION_FAILED(ErrorCategory.CONVERSION, 500, "CONVERSION_004", "Converted model validation failed",
            false),
    QUANTIZATION_FAILED(ErrorCategory.CONVERSION, 500, "CONVERSION_005", "Model quantization failed", true),

    // ===== Request Validation Errors (400) =====
    VALIDATION_MISSING_FIELD(ErrorCategory.VALIDATION, 400, "VALIDATION_001", "Required field missing", false),
    VALIDATION_INVALID_FORMAT(ErrorCategory.VALIDATION, 400, "VALIDATION_002", "Invalid field format", false),
    VALIDATION_CONSTRAINT_VIOLATION(ErrorCategory.VALIDATION, 400, "VALIDATION_003", "Validation constraint violated",
            false),

    // ===== Circuit Breaker & Resilience (503) =====
    CIRCUIT_BREAKER_OPEN(ErrorCategory.CIRCUIT, 503, "CIRCUIT_001", "Circuit breaker open", true),
    ALL_RUNNERS_FAILED(ErrorCategory.CIRCUIT, 503, "CIRCUIT_002", "All runner attempts failed", true),
    FALLBACK_FAILED(ErrorCategory.CIRCUIT, 503, "CIRCUIT_003", "Fallback execution failed", true),

    // ===== Provider Errors (400, 429, 502, 503, 504) =====
    PROVIDER_NOT_INITIALIZED(ErrorCategory.PROVIDER, 503, "PROVIDER_001", "Provider not initialized", true),
    PROVIDER_UNAVAILABLE(ErrorCategory.PROVIDER, 503, "PROVIDER_002", "Provider unavailable", true),
    PROVIDER_TIMEOUT(ErrorCategory.PROVIDER, 504, "PROVIDER_003", "Provider timeout", true),
    PROVIDER_AUTH_FAILED(ErrorCategory.PROVIDER, 401, "PROVIDER_004", "Provider authentication failed", false),
    PROVIDER_RATE_LIMITED(ErrorCategory.PROVIDER, 429, "PROVIDER_005", "Provider rate limit exceeded", true),
    PROVIDER_QUOTA_EXCEEDED(ErrorCategory.PROVIDER, 429, "PROVIDER_006", "Provider quota exceeded", false),
    PROVIDER_BAD_RESPONSE(ErrorCategory.PROVIDER, 502, "PROVIDER_007", "Provider returned invalid response", true),
    PROVIDER_STREAM_FAILED(ErrorCategory.PROVIDER, 502, "PROVIDER_008", "Provider stream failed", true),
    PROVIDER_INVALID_REQUEST(ErrorCategory.PROVIDER, 400, "PROVIDER_009", "Provider request invalid", false),
    PROVIDER_INIT_FAILED(ErrorCategory.PROVIDER, 500, "PROVIDER_010", "Provider initialization failed", false),

    // ===== Routing & Registry Errors (403, 404, 503) =====
    ROUTING_NO_COMPATIBLE_PROVIDER(ErrorCategory.ROUTING, 503, "ROUTING_001", "No compatible provider available", true),
    ROUTING_PROVIDER_NOT_FOUND(ErrorCategory.ROUTING, 404, "ROUTING_002", "Provider not found", false),
    ROUTING_POLICY_REJECTED(ErrorCategory.ROUTING, 403, "ROUTING_003", "Routing policy rejected request", false),

    // ===== Plugin Errors (500, 504) =====
    PLUGIN_INITIALIZATION_FAILED(ErrorCategory.PLUGIN, 500, "PLUGIN_001", "Plugin initialization failed", false),
    PLUGIN_EXECUTION_FAILED(ErrorCategory.PLUGIN, 500, "PLUGIN_002", "Plugin execution failed", true),
    PLUGIN_INVALID_CONFIGURATION(ErrorCategory.PLUGIN, 500, "PLUGIN_003", "Plugin configuration invalid", false),

    // ===== Configuration Errors (400, 500) =====
    CONFIG_MISSING(ErrorCategory.CONFIG, 500, "CONFIG_001", "Required configuration missing", false),
    CONFIG_INVALID(ErrorCategory.CONFIG, 500, "CONFIG_002", "Invalid configuration value", false),
    CONFIG_UNSUPPORTED(ErrorCategory.CONFIG, 400, "CONFIG_003", "Unsupported configuration", false),

    // ===== Network Errors (502, 503, 504) =====
    NETWORK_TIMEOUT(ErrorCategory.NETWORK, 504, "NETWORK_001", "Network timeout", true),
    NETWORK_UNREACHABLE(ErrorCategory.NETWORK, 503, "NETWORK_002", "Network unreachable", true),
    NETWORK_DNS_FAILED(ErrorCategory.NETWORK, 503, "NETWORK_003", "DNS resolution failed", true),
    NETWORK_TLS_FAILED(ErrorCategory.NETWORK, 502, "NETWORK_004", "TLS handshake failed", false),
    NETWORK_PROTOCOL_ERROR(ErrorCategory.NETWORK, 502, "NETWORK_005", "Network protocol error", true),
    NETWORK_BAD_RESPONSE(ErrorCategory.NETWORK, 502, "NETWORK_006", "Network response invalid", true),

    // ===== Stream Errors (500, 502, 504) =====
    STREAM_INIT_FAILED(ErrorCategory.STREAM, 500, "STREAM_001", "Stream initialization failed", false),
    STREAM_DISCONNECTED(ErrorCategory.STREAM, 502, "STREAM_002", "Stream disconnected", true),
    STREAM_PROTOCOL_ERROR(ErrorCategory.STREAM, 502, "STREAM_003", "Stream protocol error", true),
    STREAM_TIMEOUT(ErrorCategory.STREAM, 504, "STREAM_004", "Stream timeout", true),

    // ===== Internal Errors (500) =====
    INTERNAL_ERROR(ErrorCategory.INTERNAL, 500, "INTERNAL_001", "Internal server error", true),
    DATABASE_ERROR(ErrorCategory.INTERNAL, 500, "INTERNAL_002", "Database error", true),
    CACHE_ERROR(ErrorCategory.INTERNAL, 500, "INTERNAL_003", "Cache error", true),
    SERIALIZATION_ERROR(ErrorCategory.INTERNAL, 500, "INTERNAL_004", "Serialization error", true);

    public enum ErrorCategory {
        MODEL("MODEL"),
        TENSOR("TENSOR"),
        DEVICE("DEVICE"),
        QUOTA("QUOTA"),
        AUTH("AUTH"),
        INIT("INIT"),
        RUNTIME("RUNTIME"),
        STORAGE("STORAGE"),
        CONVERSION("CONVERSION"),
        VALIDATION("VALIDATION"),
        CIRCUIT("CIRCUIT"),
        PROVIDER("PROVIDER"),
        ROUTING("ROUTING"),
        PLUGIN("PLUGIN"),
        CONFIG("CONFIG"),
        NETWORK("NETWORK"),
        STREAM("STREAM"),
        INTERNAL("INTERNAL");

        private final String prefix;

        ErrorCategory(String prefix) {
            this.prefix = prefix;
        }

        public String getPrefix() {
            return prefix;
        }
    }

    private static final Map<String, ErrorCode> BY_CODE;
    private static final Map<String, ErrorCode> ALIASES;
    private static final Map<ErrorCategory, Map<String, ErrorCode>> BY_CATEGORY;

    static {
        Map<String, ErrorCode> byCode = new HashMap<>();
        Map<String, ErrorCode> aliases = new HashMap<>();
        Map<ErrorCategory, Map<String, ErrorCode>> byCategory = new EnumMap<>(ErrorCategory.class);

        for (ErrorCategory category : ErrorCategory.values()) {
            byCategory.put(category, new HashMap<>());
        }

        for (ErrorCode errorCode : values()) {
            if (!errorCode.code.startsWith(errorCode.category.getPrefix() + "_")) {
                throw new IllegalStateException(
                        "ErrorCode prefix mismatch: " + errorCode.name() + " -> " + errorCode.code);
            }
            ErrorCode previous = byCode.put(errorCode.code, errorCode);
            if (previous != null) {
                throw new IllegalStateException(
                        "Duplicate error code: " + errorCode.code + " (" + errorCode.name() + ", " + previous.name()
                                + ")");
            }
            byCategory.get(errorCode.category).put(errorCode.code, errorCode);
        }

        // Legacy or external aliases mapped to canonical codes
        aliases.put("PROVIDER_UNAVAILABLE", PROVIDER_UNAVAILABLE);
        aliases.put("PROVIDER_TIMEOUT", PROVIDER_TIMEOUT);
        aliases.put("PROVIDER_RATE_LIMIT_EXCEEDED", PROVIDER_RATE_LIMITED);
        aliases.put("PROVIDER_QUOTA_EXCEEDED", PROVIDER_QUOTA_EXCEEDED);
        aliases.put("PROVIDER_AUTH_FAILED", PROVIDER_AUTH_FAILED);
        aliases.put("PROVIDER_ERROR", PROVIDER_BAD_RESPONSE);
        aliases.put("AUTH_FAILED", PROVIDER_AUTH_FAILED);
        aliases.put("RATE_LIMIT_EXCEEDED", RATE_LIMIT_EXCEEDED);
        aliases.put("QUOTA_EXCEEDED", QUOTA_EXCEEDED);
        aliases.put("NO_COMPATIBLE_PROVIDER", ROUTING_NO_COMPATIBLE_PROVIDER);

        BY_CODE = Collections.unmodifiableMap(byCode);
        ALIASES = Collections.unmodifiableMap(aliases);

        Map<ErrorCategory, Map<String, ErrorCode>> immutableByCategory = new EnumMap<>(ErrorCategory.class);
        for (Map.Entry<ErrorCategory, Map<String, ErrorCode>> entry : byCategory.entrySet()) {
            immutableByCategory.put(entry.getKey(), Collections.unmodifiableMap(entry.getValue()));
        }
        BY_CATEGORY = Collections.unmodifiableMap(immutableByCategory);
    }

    private final ErrorCategory category;
    private final int httpStatus;
    private final String code;
    private final String defaultMessage;
    private final boolean retryable;

    ErrorCode(ErrorCategory category, int httpStatus, String code, String defaultMessage, boolean retryable) {
        this.category = Objects.requireNonNull(category, "category");
        this.httpStatus = httpStatus;
        this.code = Objects.requireNonNull(code, "code");
        this.defaultMessage = Objects.requireNonNull(defaultMessage, "defaultMessage");
        this.retryable = retryable;
    }

    /**
     * Get ErrorCode from code string.
     * Returns INTERNAL_ERROR when the code is unknown.
     */
    public static ErrorCode fromCode(String code) {
        return fromCode(code, INTERNAL_ERROR);
    }

    /**
     * Get ErrorCode from code string with a custom fallback.
     */
    public static ErrorCode fromCode(String code, ErrorCode fallback) {
        if (code == null || code.isBlank()) {
            return fallback;
        }
        String trimmed = code.trim();
        ErrorCode errorCode = BY_CODE.get(trimmed);
        if (errorCode != null) {
            return errorCode;
        }
        ErrorCode alias = ALIASES.get(trimmed.toUpperCase(Locale.ROOT));
        return alias != null ? alias : fallback;
    }

    /**
     * Returns an immutable map of codes within a category.
     */
    public static Map<String, ErrorCode> byCategory(ErrorCategory category) {
        Map<String, ErrorCode> codes = BY_CATEGORY.get(category);
        return codes == null ? Collections.emptyMap() : codes;
    }

    public ErrorCategory getCategory() {
        return category;
    }

    public boolean isClientError() {
        return httpStatus >= 400 && httpStatus < 500;
    }

    public boolean isServerError() {
        return httpStatus >= 500 && httpStatus < 600;
    }

    public int getHttpStatus() {
        return httpStatus;
    }

    public String getCode() {
        return code;
    }

    public String getDefaultMessage() {
        return defaultMessage;
    }

    public boolean isRetryable() {
        return retryable;
    }
}



================================================
FILE: core/tafkir-error-code/src/main/java/tech/kayys/tafkir/error/ErrorCodeDoc.java
================================================
package tech.kayys.tafkir.error;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility to generate Markdown documentation for {@link ErrorCode}.
 */
public final class ErrorCodeDoc {

    private ErrorCodeDoc() {
    }

    public static String toMarkdown() {
        String nl = System.lineSeparator();
        StringBuilder builder = new StringBuilder();
        builder.append("# Tafkir Error Codes").append(nl).append(nl);
        builder.append("Generated from `ErrorCode` at build time.").append(nl).append(nl);

        for (ErrorCode.ErrorCategory category : ErrorCode.ErrorCategory.values()) {
            List<ErrorCode> codes = byCategoryInOrder(category);
            if (codes.isEmpty()) {
                continue;
            }
            builder.append("## ").append(category.getPrefix()).append(nl).append(nl);
            builder.append("| Code | HTTP | Retryable | Message |").append(nl);
            builder.append("| --- | --- | --- | --- |").append(nl);
            for (ErrorCode code : codes) {
                builder.append("| ")
                        .append(code.getCode()).append(" | ")
                        .append(code.getHttpStatus()).append(" | ")
                        .append(code.isRetryable()).append(" | ")
                        .append(code.getDefaultMessage())
                        .append(" |").append(nl);
            }
            builder.append(nl);
        }

        return builder.toString();
    }

    public static void main(String[] args) {
        System.out.print(toMarkdown());
    }

    private static List<ErrorCode> byCategoryInOrder(ErrorCode.ErrorCategory category) {
        List<ErrorCode> codes = new ArrayList<>();
        for (ErrorCode errorCode : ErrorCode.values()) {
            if (errorCode.getCategory() == category) {
                codes.add(errorCode);
            }
        }
        return codes;
    }
}



================================================
FILE: core/tafkir-error-code/src/main/java/tech/kayys/tafkir/error/ErrorPayload.java
================================================
package tech.kayys.tafkir.error;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Standardized error payload for all inference failures.
 * Integrates with tafkir's error-as-input pattern.
 */
public final class ErrorPayload {

    @NotBlank
    private final String type;

    @NotBlank
    private final String message;

    private final Map<String, Object> details;
    private final boolean retryable;
    private final String originNode;
    private final String originRunId;
    private final int attempt;
    private final int maxAttempts;

    @NotNull
    private final Instant timestamp;

    private final String suggestedAction;
    private final String provenanceRef;

    @JsonCreator
    public ErrorPayload(
            @JsonProperty("type") String type,
            @JsonProperty("message") String message,
            @JsonProperty("details") Map<String, Object> details,
            @JsonProperty("retryable") boolean retryable,
            @JsonProperty("originNode") String originNode,
            @JsonProperty("originRunId") String originRunId,
            @JsonProperty("attempt") int attempt,
            @JsonProperty("maxAttempts") int maxAttempts,
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("suggestedAction") String suggestedAction,
            @JsonProperty("provenanceRef") String provenanceRef) {
        this.type = Objects.requireNonNull(type, "type");
        this.message = Objects.requireNonNull(message, "message");
        this.details = details != null
                ? Collections.unmodifiableMap(new HashMap<>(details))
                : Collections.emptyMap();
        this.retryable = retryable;
        this.originNode = originNode;
        this.originRunId = originRunId;
        this.attempt = attempt;
        this.maxAttempts = maxAttempts;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.suggestedAction = suggestedAction;
        this.provenanceRef = provenanceRef;
    }

    // Getters
    public String getType() {
        return type;
    }

    public String getMessage() {
        return message;
    }

    public Map<String, Object> getDetails() {
        return details;
    }

    public boolean isRetryable() {
        return retryable;
    }

    public String getOriginNode() {
        return originNode;
    }

    public String getOriginRunId() {
        return originRunId;
    }

    public int getAttempt() {
        return attempt;
    }

    public int getMaxAttempts() {
        return maxAttempts;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public String getSuggestedAction() {
        return suggestedAction;
    }

    public String getProvenanceRef() {
        return provenanceRef;
    }

    // Factory methods
    public static ErrorPayload from(Throwable error, String nodeId, String runId) {
        return builder()
                .type(error.getClass().getSimpleName())
                .message(error.getMessage() != null ? error.getMessage() : "Unknown error")
                .originNode(nodeId)
                .originRunId(runId)
                .retryable(isRetryable(error))
                .suggestedAction(determineSuggestedAction(error))
                .details(extractDetails(error))
                .build();
    }

    private static boolean isRetryable(Throwable error) {
        String className = error.getClass().getName();
        return !className.contains("Validation") &&
                !className.contains("Authorization") &&
                !className.contains("Quota");
    }

    private static String determineSuggestedAction(Throwable error) {
        String className = error.getClass().getName();
        if (className.contains("Quota")) {
            return "escalate";
        } else if (className.contains("Provider")) {
            return "retry";
        } else if (className.contains("Validation")) {
            return "human_review";
        } else {
            return "fallback";
        }
    }

    private static Map<String, Object> extractDetails(Throwable error) {
        Map<String, Object> details = new HashMap<>();
        details.put("errorClass", error.getClass().getName());
        if (error.getCause() != null) {
            details.put("cause", error.getCause().getMessage());
        }
        return details;
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String type;
        private String message;
        private final Map<String, Object> details = new HashMap<>();
        private boolean retryable = false;
        private String originNode;
        private String originRunId;
        private int attempt = 0;
        private int maxAttempts = 3;
        private Instant timestamp = Instant.now();
        private String suggestedAction;
        private String provenanceRef;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder message(String message) {
            this.message = message;
            return this;
        }

        public Builder detail(String key, Object value) {
            if (value != null) {
                this.details.put(key, value);
            }
            return this;
        }

        public Builder details(Map<String, Object> details) {
            this.details.putAll(details);
            return this;
        }

        public Builder retryable(boolean retryable) {
            this.retryable = retryable;
            return this;
        }

        public Builder originNode(String originNode) {
            this.originNode = originNode;
            return this;
        }

        public Builder originRunId(String originRunId) {
            this.originRunId = originRunId;
            return this;
        }

        public Builder attempt(int attempt) {
            this.attempt = attempt;
            return this;
        }

        public Builder maxAttempts(int maxAttempts) {
            this.maxAttempts = maxAttempts;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder suggestedAction(String suggestedAction) {
            this.suggestedAction = suggestedAction;
            return this;
        }

        public Builder provenanceRef(String provenanceRef) {
            this.provenanceRef = provenanceRef;
            return this;
        }

        public ErrorPayload build() {
            Objects.requireNonNull(type, "type is required");
            Objects.requireNonNull(message, "message is required");
            return new ErrorPayload(
                    type, message, details, retryable, originNode, originRunId,
                    attempt, maxAttempts, timestamp, suggestedAction, provenanceRef);
        }
    }

    @Override
    public String toString() {
        return "ErrorPayload{" +
                "type='" + type + '\'' +
                ", message='" + message + '\'' +
                ", retryable=" + retryable +
                ", attempt=" + attempt +
                ", suggestedAction='" + suggestedAction + '\'' +
                '}';
    }
}


================================================
FILE: examples/README.md
================================================
# Tafkir Examples

A comprehensive collection of working examples demonstrating Tafkir's capabilities across JBang scripts and Jupyter notebooks.

## 📚 Example Categories

### 🎨 **Multimodal AI** (Image, Audio, Video)

#### JBang Examples
- [stable_diffusion_generation.java](jbang/multimodal/stable_diffusion_generation.java) - Text-to-image generation with Stable Diffusion ✨ NEW
- [stable_diffusion_batch.java](jbang/multimodal/stable_diffusion_batch.java) - Batch generate multiple variations ✨ NEW
- [stable_diffusion_metal.java](jbang/multimodal/stable_diffusion_metal.java) - SD with Metal acceleration (Apple Silicon)
- [text_to_speech.java](jbang/multimodal/text_to_speech.java) - Text-to-Speech with SpeechT5 ✨ NEW
- [vision_storyteller.java](jbang/multimodal/vision_storyteller.java) - Image-to-text with LLaVA
- [native_whisper_stt.java](jbang/multimodal/native_whisper_stt.java) - Speech-to-Text with Whisper
- [smart_transcriber.java](jbang/multimodal/smart_transcriber.java) - Audio transcription + highlights
- [video_analyst.java](jbang/multimodal/video_analyst.java) - Video temporal analysis
- [omni_assistant.java](jbang/multimodal/omni_assistant.java) - Mixed-modality reasoning

#### Jupyter Notebooks
- [07-stable-diffusion.ipynb](jupyter/07-stable-diffusion.ipynb) - Interactive SD generation with visualization ✨ NEW

### 🗣️ **Natural Language Processing**

#### JBang Examples
- [nlp_chat_qwen_gguf.java](jbang/nlp/nlp_chat_qwen_gguf.java) - Chat with Qwen GGUF model
- [nlp_concurrent_batch_pipeline.java](jbang/nlp/nlp_concurrent_batch_pipeline.java) - High-throughput batching
- [nlp_high_performance.java](jbang/nlp/nlp_high_performance.java) - Optimized text generation
- [nlp_semantic_search_simd.java](jbang/nlp/nlp_semantic_search_simd.java) - Vector similarity search
- [nlp_sentiment_analysis.java](jbang/nlp/nlp_sentiment_analysis.java) - Text classification
- [nlp_transformer_classifier.java](jbang/nlp/nlp_transformer_classifier.java) - Custom classifier training

### 🔧 **ML Framework & Compatibility**

#### JBang Examples
- [tafkir-quickstart.java](jbang/sdk/tafkir-quickstart.java) - Quick start guide for the ML framework migration
- [tafkir-sdk-core-example.java](jbang/sdk/tafkir-sdk-core-example.java) - Legacy SDK-named tensor/core compatibility sample
- [tafkir-sdk-vision-example.java](jbang/sdk/tafkir-sdk-vision-example.java) - Legacy SDK-named vision compatibility sample
- [tafkir-sdk-train-example.java](jbang/sdk/tafkir-sdk-train-example.java) - Legacy SDK-named trainer compatibility sample
- [tafkir-sdk-augment-example.java](jbang/sdk/tafkir-sdk-augment-example.java) - Legacy SDK-named augmentation compatibility sample
- [tafkir-sdk-export-example.java](jbang/sdk/tafkir-sdk-export-example.java) - Legacy SDK-named export compatibility sample
- [graph_fusion_example.java](jbang/sdk/graph_fusion_example.java) - Graph optimization
- [unified_framework_demo.java](jbang/sdk/unified_framework_demo.java) - Unified inference
- [mnist_training_v02.java](jbang/sdk/mnist_training_v02.java) - MNIST training
- [pytorch_comparison_v02.java](jbang/sdk/pytorch_comparison_v02.java) - PyTorch comparison
- [tensor_operations_v02.java](jbang/sdk/tensor_operations_v02.java) - Tensor math
- [tokenization_v02.java](jbang/sdk/tokenization_v02.java) - Tokenizer usage
- [vision_transforms_v02.java](jbang/sdk/vision_transforms_v02.java) - Image transforms

### ⚡ **Edge Inference (LiteRT)**

#### JBang Examples
- [LiteRTEdgeExample.java](jbang/edge/LiteRTEdgeExample.java) - Basic LiteRT usage
- [ImageClassificationExample.java](jbang/edge/ImageClassificationExample.java) - Image classification
- [ObjectDetectionExample.java](jbang/edge/ObjectDetectionExample.java) - Object detection
- [TextGenerationExample.java](jbang/edge/TextGenerationExample.java) - On-device text generation
- [BenchmarkExample.java](jbang/edge/BenchmarkExample.java) - Performance benchmarking
- [LiteRTMetalBridgeExample.java](jbang/edge/LiteRTMetalBridgeExample.java) - Metal acceleration
- [LiteRTGemmaExample.java](jbang/edge/LiteRTGemmaExample.java) - Gemma on edge
- [LiteRTGemmaMetalExample.java](jbang/edge/LiteRTGemmaMetalExample.java) - Gemma + Metal

### 🔢 **Quantization**

#### JBang Examples
- [tafkir-quantizer-gptq.java](jbang/quantizer/tafkir-quantizer-gptq.java) - GPTQ quantization
- [tafkir-quantizer-awq.java](jbang/quantizer/tafkir-quantizer-awq.java) - AWQ quantization
- [tafkir-quantizer-autoround.java](jbang/quantizer/tafkir-quantizer-autoround.java) - AutoRound quantization
- [tafkir-quantizer-turboquant.java](jbang/quantizer/tafkir-quantizer-turboquant.java) - TurboQuant (INT4/INT8/FP8)
- [tafkir-quantizer-comparison.java](jbang/quantizer/tafkir-quantizer-comparison.java) - Method comparison

### 🧠 **Common/Basic**

#### JBang Examples
- [hello_tafkir.java](jbang/common/hello_tafkir.java) - Hello World
- [error_handling.java](jbang/common/error_handling.java) - Error handling patterns
- [batch_process.java](jbang/common/batch_process.java) - Batch processing
- [custom_module_demo.java](jbang/common/custom_module_demo.java) - Custom modules
- [mnist_style_setup.java](jbang/common/mnist_style_setup.java) - Simple training
- [model_persistence.java](jbang/common/model_persistence.java) - Save/load models
- [model_persistence_safetensor.java](jbang/common/model_persistence_safetensor.java) - SafeTensors format
- [train_cli.java](jbang/common/train_cli.java) - Training CLI
- [train_model.java](jbang/common/train_model.java) - Model training

### 🔌 **Third-Party Integrations**

#### JBang Examples
- [wayang_tafkir_serving_bridge.java](jbang/integration/wayang_tafkir_serving_bridge.java) - Wayang-Tafkir agent runtime on top of Tafkir serving APIs
- [deeplearning4j_integration.java](jbang/integration/deeplearning4j_integration.java) - DL4J integration
- [opennlp_integration.java](jbang/integration/opennlp_integration.java) - Apache OpenNLP
- [smile_ml_integration.java](jbang/integration/smile_ml_integration.java) - Smile ML library
- [stanford_nlp_integration.java](jbang/integration/stanford_nlp_integration.java) - Stanford NLP
- [tribuo_integration.java](jbang/integration/tribuo_integration.java) - Oracle Tribuo

### 🧪 **Neural Networks & ML**

#### JBang Examples
- [neural_network_with_tafkir.java](jbang/neural_network/neural_network_with_tafkir.java) - Custom NN
- [neural-network-example.java](jbang/neural_network/neural-network-example.java) - Simple NN
- [cv_mlp_classifier_export.java](jbang/machine_learning/cv_mlp_classifier_export.java) - MLP classifier
- [simd_audio_processing.java](jbang/machine_learning/simd_audio_processing.java) - SIMD audio

## 🚀 Quick Start

### JBang (Command-Line)

```bash
# Install JBang first
curl -Ls https://sh.jbang.dev | bash -s - app setup

# Run any example
jbang examples/jbang/multimodal/stable_diffusion_generation.java

# With custom parameters
jbang examples/jbang/multimodal/stable_diffusion_generation.java \
  --prompt "a cyberpunk cat" \
  --output cyberpunk.png \
  --steps 30
```

### Jupyter Notebook (Interactive)

```bash
# Install Java kernel
jbang app install java-kernel

# Register kernel
jupyter kernelspec install examples/jupyter/kernel.json

# Launch Jupyter
jupyter notebook examples/jupyter

# Open 07-stable-diffusion.ipynb for image generation
```

## 📊 Feature Matrix

| Category | JBang | Jupyter | Description |
|----------|-------|---------|-------------|
| **Text-to-Image** | ✅ 3 | ✅ 1 | Stable Diffusion generation |
| **Text-to-Speech** | ✅ 1 | ❌ | SpeechT5 synthesis |
| **Speech-to-Text** | ✅ 2 | ❌ | Whisper transcription |
| **Vision QA** | ✅ 1 | ❌ | LLaVA image understanding |
| **Video Analysis** | ✅ 1 | ❌ | Temporal video understanding |
| **NLP/Chat** | ✅ 6 | ✅ 1 | Text generation & classification |
| **ML Framework / Compatibility** | ✅ 13 | ✅ 1 | Core API and migration examples |
| **Edge/LiteRT** | ✅ 9 | ❌ | Mobile/edge inference |
| **Quantization** | ✅ 5 | ❌ | Model compression |
| **Integrations** | ✅ 6 | ❌ | Third-party libraries and agent bridges |

**Total**: 54+ JBang scripts, 2+ Jupyter notebooks

## 🎯 Learning Paths

### Beginner Path
1. `hello_tafkir.java` - Basic usage
2. `01-getting-started.ipynb` - Interactive tensors
3. `mnist_style_setup.java` - Simple training
4. `tafkir-quickstart.java` - SDK basics

### Intermediate Path
1. `stable_diffusion_generation.java` - Image generation ✨ NEW
2. `native_whisper_stt.java` - Audio processing
3. `tafkir-sdk-vision-example.java` - Computer vision
4. `nlp_sentiment_analysis.java` - NLP basics

### Advanced Path
1. `omni_assistant.java` - Multimodal AI
2. `tafkir-quantizer-turboquant.java` - Quantization
3. `video_analyst.java` - Video understanding
4. `wayang_tafkir_serving_bridge.java` - Agent serving integration boundary
5. `graph_fusion_example.java` - Optimization

## 📝 Creating Your Own Examples

### Converting Jupyter → JBang

1. Extract code cells from `.ipynb`
2. Add JBang header with dependencies
3. Add CLI argument parsing
4. Run with `jbang your_script.java`

### Converting JBang → Jupyter

1. Remove JBang header
2. Split into logical code cells
3. Add markdown explanation cells
4. Save as `.ipynb`

## 🐛 Troubleshooting

### JBang Issues

**Problem**: `Error: Could not find or load main class`
**Solution**: Ensure Java 25 is installed: `java -version`

**Problem**: Dependencies not resolving
**Solution**: Run `jbang cache clear` and retry

### Jupyter Issues

**Problem**: Kernel not found
**Solution**: Install Java kernel: `jbang app install java-kernel`

**Problem**: Imports failing
**Solution**: Ensure Tafkir JARs are in kernel classpath

## 🤝 Contributing

We welcome new examples! Please follow these guidelines:

1. **Working Code**: Examples must compile and run
2. **Clear Comments**: Explain what each section does
3. **CLI Arguments**: Support `--help` and custom parameters
4. **Error Handling**: Show best practices for error handling
5. **Documentation**: Update this INDEX.md when adding new examples

## 📚 Additional Resources

- [Stable Diffusion Tutorial](https://tafkir-ai.github.io/docs/tutorials/intermediate/stable-diffusion/)
- [API Reference](https://tafkir-ai.github.io/docs/api-references/0.1.0-SNAPSHOT/modules/)
- [CLI Documentation](https://tafkir-ai.github.io/docs/cli/)
- [Framework Guides](https://tafkir-ai.github.io/docs/framework/)

---

**Last Updated**: April 14, 2026
**Total Examples**: 54+ JBang, 2+ Jupyter



================================================
FILE: examples/WORKING_STATUS.md
================================================
# Working Examples - Status & Usage

## ✅ What Works

### 1. CLI (Recommended - Full Featured)

```bash
java -jar tafkir/ui/tafkir-cli/target/tafkir-runner.jar run \
  --model CompVis/stable-diffusion-v1-4 \
  --branch onnx \
  --prompt "a cat playing ball" \
  --steps 10 \
  --output cat.png
```

**Features**: All parameters, auto-open, progress display, error handling

### 2. Shell Script (Quick Usage)

```bash
cd tafkir/examples/jbang/multimodal

# Basic
./sd_generate.sh "a cat" 10 cat.png

# Advanced
./sd_generate.sh "cyberpunk city" 30 city.png 123 8.0
# Usage: ./sd_generate.sh <prompt> <steps> <output> <seed> <cfg>
```

**Features**: Wrapper around CLI, simple interface

### 3. Jupyter Notebook (Interactive - NEEDS FIX)

**Status**: Kernel configured with Maven dependencies but needs Jupyter restart.

**To use**:
1. Stop Jupyter: `Ctrl+C`
2. Restart: `jupyter notebook tafkir/examples/jupyter/`
3. Open `07-stable-diffusion.ipynb`
4. Select kernel: `Kernel → Change kernel → Tafkir ML (Java 25)`
5. Run cells one by one

**Note**: The kernel now uses Maven coordinates (not file:// paths) which should resolve correctly.

### 4. JBang Scripts (NOT WORKING - Quarkus Extension Issue)

**Problem**: The `tafkir-runner-onnx` module is a Quarkus extension, not a standalone JAR. JBang can't run it directly without the Quarkus runtime.

**Status**: 
- ❌ `sd_direct.java` - Can't instantiate Quarkus beans outside CDI
- ❌ `stable_diffusion_generation.java` - SDK needs Quarkus CDI container
- ✅ **Shell wrapper** (`sd_generate.sh`) - Works because it calls the CLI uber-JAR

## 📋 Summary Table

| Method | Status | Ease | Features |
|--------|--------|------|----------|
| **CLI** | ✅ Working | Medium | All features |
| **Shell Script** | ✅ Working | Easy | Simplified CLI |
| **Jupyter** | ⚠️ Needs restart | Easy | Interactive |
| **JBang Direct** | ❌ Quarkus issue | Hard | N/A |

## 🚀 Recommended Workflow

### For Quick Testing
```bash
# Use the shell script
cd tafkir/examples/jbang/multimodal
./sd_generate.sh "your prompt" 10 output.png
```

### For Production/Integration
```bash
# Use CLI directly with full control
java -jar tafkir/ui/tafkir-cli/target/tafkir-runner.jar run \
  --model CompVis/stable-diffusion-v1-4 \
  --branch onnx \
  --prompt "your prompt" \
  --seed 42 \
  --steps 30 \
  --guidance-scale 7.5 \
  --output result.png
```

### For Interactive Learning (After Jupyter Restart)
```bash
# Stop current Jupyter
# Restart with:
jupyter notebook tafkir/examples/jupyter/

# Then open 07-stable-diffusion.ipynb
# Select "Tafkir ML (Java 25)" kernel
```

## 🔧 Why JBang Doesn't Work Directly

The `StableDiffusionOnnxRunner` is a **Quarkus CDI bean**:
```java
@ApplicationScoped  // Requires CDI container
public class StableDiffusionOnnxRunner extends AbstractTafkirRunner {
    @Inject  // CDI injection
    Tokenizer tokenizer;
    // ...
}
```

To run it outside Quarkus, you'd need to:
1. Manually initialize CDI container (complex)
2. Or create a standalone version without CDI (duplicate code)

**Solution**: Use the CLI which already has Quarkus runtime initialized.

## 📝 Jupyter Kernel Fix Applied

**Changed**: From `file:///path/to/jar` to Maven coordinates

**Before**:
```json
"JJAVA_DEPS": "file:///Users/bhangun/.tafkir/jupyter/tafkir-sdk-core-0.1.0-SNAPSHOT.jar,..."
```

**After**:
```json
"JJAVA_DEPS": "tech.kayys.tafkir:tafkir-sdk-java-local:0.1.0-SNAPSHOT,tech.kayys.tafkir:tafkir-runner-onnx:0.1.0-SNAPSHOT,..."
```

**Benefit**: Maven resolves transitive dependencies automatically.

## ✅ Verified Working

```bash
# Test the CLI (proven working)
java -jar tafkir/ui/tafkir-cli/target/tafkir-runner.jar run \
  --model CompVis/stable-diffusion-v1-4 \
  --branch onnx \
  --prompt "a cat" \
  --steps 3 \
  --output test.png

# Expected output:
# ✓ Image saved to: test.png
# File size: ~600-700KB
# Format: PNG 512x512
```

---

**Last Updated**: April 14, 2026  
**Working Methods**: CLI, Shell Script  
**Needs Restart**: Jupyter Notebook  
**Not Working**: Direct JBang (Quarkus limitation)



================================================
FILE: examples/docs/README.md
================================================
# Tafkir SDK Integration Examples

This directory contains comprehensive examples demonstrating how to integrate the Tafkir SDK with popular Java ML/NLP libraries.

## Available Integrations

### 1. Deeplearning4j Integration
**File**: `examples/integration/deeplearning4j_integration.java`

Integrates Tafkir SDK with [Deeplearning4j](https://deeplearning4j.konduit.ai/), a comprehensive deep learning library for the JVM.

**Use Cases**:
- Using DL4J pre-trained models (ResNet, VGG, etc.) alongside Tafkir models
- Converting between DL4J INDArrays and Tafkir tensors
- Combining DL4J's image preprocessing pipeline with Tafkir inference
- Hybrid workflows leveraging both frameworks' strengths

**Run**:
```bash
jbang examples/integration/deeplearning4j_integration.java
```

### 2. Stanford NLP Integration
**File**: `examples/integration/stanford_nlp_integration.java`

Integrates Tafkir SDK with [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/), a leading natural language processing toolkit.

**Use Cases**:
- Using Stanford NLP for advanced text preprocessing (tokenization, POS tagging, NER)
- Feeding linguistically-rich features into Tafkir models
- Combining Stanford sentiment analysis with Tafkir predictions
- Building multi-stage NLP pipelines

**Run**:
```bash
jbang examples/integration/stanford_nlp_integration.java
```

### 3. Apache OpenNLP Integration
**File**: `examples/integration/opennlp_integration.java`

Integrates Tafkir SDK with [Apache OpenNLP](https://opennlp.apache.org/), a machine learning based toolkit for processing natural language text.

**Use Cases**:
- Text preprocessing and feature extraction
- Named entity recognition
- Sentence detection and tokenization
- Combining OpenNLP tools with Tafkir inference

**Run**:
```bash
jbang examples/integration/opennlp_integration.java
```

### 4. Smile ML Integration
**File**: `examples/integration/smile_ml_integration.java`

Integrates Tafkir SDK with [Smile](https://haifengl.github.io/) (Statistical Machine Intelligence & Learning Engine), a fast and comprehensive machine learning system.

**Use Cases**:
- Using Smile's data preprocessing and normalization
- Combining Smile's traditional ML algorithms with Tafkir neural networks
- Feature engineering with Smile, inference with Tafkir
- Ensemble methods combining both libraries

**Run**:
```bash
jbang examples/integration/smile_ml_integration.java
```

### 5. Oracle Tribuo Integration
**File**: `examples/integration/tribuo_integration.java`

Integrates Tafkir SDK with [Oracle Tribuo](https://tribuo.org/), a machine learning library providing classification, regression, clustering, and more.

**Use Cases**:
- Using Tribuo's feature extraction pipelines
- Combining Tribuo classifiers with Tafkir models
- Ensemble methods using both libraries
- Multi-model prediction aggregation

**Run**:
```bash
jbang examples/integration/tribuo_integration.java
```

### 6. Multimodal Integration (Native)
 **Directory**: `examples/multimodal/`
 
 Demonstrates Tafkir's native multimodal processing capabilities (Vision, Audio, Video) using the high-level fluent API.
 
 **Use Cases**:
 - **Vision Storytelling**: Image-to-Text generation.
 - **Smart Transcription**: Audio-to-Text with summarization.
 - **Video Analysis**: Temporal scene understanding.
 - **Omni Assistant**: Cross-modality reasoning.
 
 **Run**:
 ```bash
 # Vision
 jbang examples/multimodal/vision_storyteller.java [image]
 
 # Audio
 jbang examples/multimodal/smart_transcriber.java [audio]
 
 # Video
 jbang examples/multimodal/video_analyst.java [video]
 
 # Omni
 jbang examples/multimodal/omni_assistant.java [image] [audio] [prompt]
 ```
 
 ## Common Integration Patterns

### Pattern 1: Data Conversion

Most integrations require converting between different tensor/data formats:

```java
// DL4J INDArray → Tafkir Tensor
INDArray dl4jArray = ...;
float[] data = dl4jArray.toFloatVector();
GradTensor tensor = GradTensor.of(data, ...);

// Tafkir Tensor → DL4J INDArray
GradTensor tensor = ...;
INDArray dl4jArray = Nd4j.create(tensor.data(), tensor.shape());
```

### Pattern 2: Pipeline Composition

Chain preprocessing from one library with inference from another:

```java
// Stanford NLP preprocessing → Tafkir inference
String text = "...";
List<CoreLabel> tokens = stanfordTokenizer.tokenize(text);
String[] features = extractFeatures(tokens);
float[] vector = vectorize(features);
GradTensor input = GradTensor.of(vector, ...);
GradTensor output = tafkirModel.forward(input);
```

### Pattern 3: Ensemble Predictions

Combine predictions from multiple libraries:

```java
// Get predictions from both libraries
float dl4jScore = dl4jModel.predict(input);
float tafkirScore = tafkirModel.predict(input);

// Ensemble: weighted average
float ensembleScore = 0.6f * dl4jScore + 0.4f * tafkirScore;
```

### Pattern 4: Feature Fusion

Concatenate features from different sources:

```java
// Extract features from multiple sources
float[] dl4jFeatures = dl4jExtractor.extract(image);
float[] tafkirFeatures = tafkirExtractor.extract(image);

// Concatenate and feed to classifier
float[] combined = concat(dl4jFeatures, tafkirFeatures);
GradTensor input = GradTensor.of(combined, 1, combined.length);
```

## Prerequisites

### Java Version
- Java 21 or higher required for all examples

### jbang Setup
All examples use [jbang](https://jbang.dev/) for dependency management:

```bash
# Install jbang
curl -Ls https://sh.jbang.dev | bash -

# Verify installation
jbang version
```

### Network Access
First run will download dependencies (may take 1-2 minutes):
- Maven Central repositories
- Library-specific repositories (e.g., Sonatype for Stanford NLP)

## Running the Examples

### Basic Execution

```bash
cd tafkir/sdk/integration/jbang-templates/examples/integration
jbang deeplearning4j_integration.java
```

### With Custom Arguments

Some examples support command-line arguments:

```bash
jbang examples/integration/stanford_nlp_integration.java --model=sentiment --text="Great movie!"
```

### Export as Standalone JAR

```bash
jbang -o integration.jar examples/integration/deeplearning4j_integration.java
java -jar integration.jar
```

## Troubleshooting

### Slow First Run
**Normal behavior** - jbang downloads and caches dependencies.

**Solution**: Subsequent runs are fast. For offline use:
```bash
jbang cache deps examples/integration/deeplearning4j_integration.java
```

### Memory Issues
Some libraries (especially DL4J and Stanford NLP) require significant memory.

**Solution**: Add JAVA_OPTIONS in the script or use:
```bash
JAVA_OPTS="-Xmx4g" jbang deeplearning4j_integration.java
```

### Native Library Errors
Some libraries use native code (ND4J, etc.).

**Solution**: Ensure `--enable-native-access=ALL-UNNAMED` is in JAVA_OPTIONS (already included in examples).

### Dependency Conflicts
If you encounter version conflicts:

**Solution**: Clear jbang cache:
```bash
jbang cache clear
```

## Integration Best Practices

### 1. Resource Management
Always close resources properly:

```java
try (var model = loadModel()) {
    // use model
} finally {
    model.close();
}
```

### 2. Thread Safety
Most ML libraries are not thread-safe by default. Use proper synchronization:

```java
synchronized(model) {
    model.predict(input);
}
```

### 3. Memory Efficiency
Release unused tensors/arrays:

```java
// DL4J
Nd4j.getMemoryManager().garbageCollect();

// Tafkir
tensor.detach();
```

### 4. Error Handling
Wrap external library calls in try-catch:

```java
try {
    // External library call
} catch (ExternalLibraryException e) {
    // Handle gracefully
    logger.warn("External library failed, using fallback", e);
}
```

## Performance Considerations

### Warm-up
First inference is slow due to JIT compilation:

```java
// Warm-up
for (int i = 0; i < 3; i++) {
    model.forward(dummyInput);
}
```

### Batching
Process multiple inputs together for better throughput:

```java
// Batch processing
GradTensor batch = GradTensor.of(batchData, batchSize, features);
GradTensor outputs = model.forward(batch);
```

### Model Caching
Cache loaded models for reuse:

```java
static final Model CACHED_MODEL = loadModel();

static {
    // Pre-warm
    CACHED_MODEL.forward(warmupInput);
}
```

## Contributing

When adding new integration examples:

1. Follow the existing naming convention: `{library}_integration.java`
2. Include comprehensive comments
3. Show at least 2-3 use cases
4. Include error handling
5. Add to this README
6. Test with `jbang {file}.java`

## Additional Resources

- [Tafkir SDK Documentation](../../../README.md)
- [Deeplearning4j Documentation](https://deeplearning4j.konduit.ai/)
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
- [Apache OpenNLP](https://opennlp.apache.org/)
- [Smile ML](https://haifengl.github.io/)
- [Oracle Tribuo](https://tribuo.org/)
- [jbang Documentation](https://jbang.dev/documentation)

## License

These examples are provided under the same license as the Tafkir SDK.



================================================
FILE: examples/docs/INTEGRATION_GUIDE.md
================================================
# Tafkir SDK Integration Examples - Quick Start Guide

This directory contains comprehensive examples demonstrating how to integrate the Tafkir SDK with popular Java ML/NLP libraries.

## 📁 Available Examples

| Library | File | Use Case | Complexity |
|---------|------|----------|------------|
| **Deeplearning4j** | `deeplearning4j_integration.java` | Image preprocessing, transfer learning, ensemble | ⭐⭐⭐ |
| **Stanford NLP** | `stanford_nlp_integration.java` | Text preprocessing, POS tagging, NER, sentiment | ⭐⭐⭐⭐ |
| **Apache OpenNLP** | `opennlp_integration.java` | Tokenization, sentence detection, NER | ⭐⭐ |
| **Smile ML** | `smile_ml_integration.java` | Feature selection, statistical features, ensemble | ⭐⭐⭐ |
| **Oracle Tribuo** | `tribuo_integration.java` | XGBoost, text classification, regression | ⭐⭐⭐⭐ |

## 🚀 Quick Start

### Prerequisites

```bash
# Java 21+ required
java --version

# jbang installed
jbang version
```

### Running Examples

```bash
cd tafkir/sdk/integration/jbang-templates/examples/integration

# Run any example (first run downloads dependencies)
jbang deeplearning4j_integration.java
jbang stanford_nlp_integration.java
jbang opennlp_integration.java
jbang smile_ml_integration.java
jbang tribuo_integration.java
```

## 📚 Integration Patterns

### Pattern 1: External Preprocessing → Tafkir Inference

**Libraries**: Deeplearning4j, Smile, OpenNLP

```java
// External library handles preprocessing
ExternalPreprocessor preprocessor = new ExternalPreprocessor();
float[] processed = preprocessor.process(rawData);

// Tafkir handles inference
GradTensor input = GradTensor.of(processed, 1, processed.length);
GradTensor output = tafkirModel.forward(input);
```

**Use Case**: Leverage mature preprocessing pipelines from established libraries while using Tafkir for lightweight inference.

### Pattern 2: Feature Extraction → Tafkir Classification

**Libraries**: Stanford NLP, OpenNLP, Tribuo

```java
// Extract linguistic features
float[] features = nlpPipeline.extractFeatures(text);

// Classify with Tafkir
GradTensor input = GradTensor.of(features, 1, features.length);
Prediction pred = classifier.predict(input);
```

**Use Case**: Rich feature extraction from NLP libraries, neural classification with Tafkir.

### Pattern 3: Ensemble (External + Tafkir)

**Libraries**: All

```java
// Get predictions from both models
float externalScore = externalModel.predict(input);
float tafkirScore = tafkirModel.predict(input);

// Ensemble: weighted average
float ensembleScore = 0.5f * externalScore + 0.5f * tafkirScore;
```

**Use Case**: Improve accuracy by combining strengths of different frameworks.

### Pattern 4: Feature Selection → Tafkir Training

**Libraries**: Smile, Tribuo

```java
// Use external library for feature importance
int[] selectedFeatures = featureSelector.select(data, labels);

// Train Tafkir on selected features
float[][] reducedData = selectFeatures(data, selectedFeatures);
tafkirModel.train(reducedData, labels);
```

**Use Case**: Reduce dimensionality using established feature selection algorithms.

### Pattern 5: Tensor Conversion

**Libraries**: Deeplearning4j, Smile

```java
// DL4J INDArray → Tafkir GradTensor
INDArray dl4jArray = ...;
GradTensor tensor = indArrayToGradTensor(dl4jArray);

// Tafkir GradTensor → DL4J INDArray
GradTensor tensor = ...;
INDArray dl4jArray = gradTensorToIndArray(tensor);
```

**Use Case**: Seamless data exchange between frameworks.

## 📖 Detailed Examples

### Deeplearning4j Integration

**File**: `deeplearning4j_integration.java`

**Patterns Demonstrated**:
1. DL4J preprocessing → Tafkir inference
2. DL4J feature extraction → Tafkir classification
3. Ensemble (DL4J + Tafkir)
4. Bidirectional tensor conversion

**Key Classes**:
- `DL4JPreprocessorTafkirInference`
- `DL4JFeatureExtractorTafkirClassifier`
- `EnsembleDL4JTafkir`

**Run**:
```bash
jbang deeplearning4j_integration.java
```

**Dependencies**:
- deeplearning4j-core
- nd4j-native-platform
- tafkir-sdk-nn

### Stanford NLP Integration

**File**: `stanford_nlp_integration.java`

**Patterns Demonstrated**:
1. Linguistic feature extraction
2. Text classification with POS/NER features
3. Hybrid sentiment analysis (Stanford + Tafkir)
4. Multi-stage NLP pipeline

**Key Classes**:
- `LinguisticFeatureExtractor`
- `LinguisticTextClassifier`
- `HybridSentimentAnalyzer`

**Features Extracted**:
- Sentence length, avg word length
- POS tag distribution (12 tags)
- Named entity counts (7 categories)
- Stanford sentiment score
- Parse tree depth

**Run**:
```bash
jbang stanford_nlp_integration.java
```

**Dependencies**:
- stanford-corenlp (with models)
- tafkir-sdk-nn

### Apache OpenNLP Integration

**File**: `opennlp_integration.java`

**Patterns Demonstrated**:
1. Sentence detection → tokenization → POS tagging
2. Named entity recognition
3. Text classification with OpenNLP features
4. NER-based document categorization

**Key Classes**:
- `OpenNLPPipeline`
- `OpenNLPTextClassifier`
- `NERBasedClassifier`

**Features Extracted**:
- Sentence/token counts
- POS ratios (noun, verb, adjective, adverb)
- Named entity distribution
- Vocabulary richness

**Run**:
```bash
jbang opennlp_integration.java
```

**Dependencies**:
- opennlp-tools
- tafkir-sdk-nn

**Note**: Downloads OpenNLP models on first run (cached in temp directory).

### Smile ML Integration

**File**: `smile_ml_integration.java`

**Patterns Demonstrated**:
1. Smile preprocessing → Tafkir inference
2. Smile feature selection → Tafkir classification
3. Ensemble (Smile Random Forest + Tafkir NN)
4. Statistical feature extraction

**Key Classes**:
- `SmilePreprocessorTafkirModel`
- `SmileFeatureSelectionTafkirClassifier`
- `EnsembleSmileTafkir`
- `StatisticalFeatureExtractor`

**Statistical Features**:
- Mean, variance, standard deviation
- Min, max, skewness, kurtosis

**Run**:
```bash
jbang smile_ml_integration.java
```

**Dependencies**:
- smile-core, smile-data, smile-math
- tafkir-sdk-nn

### Oracle Tribuo Integration

**File**: `tribuo_integration.java`

**Patterns Demonstrated**:
1. Tribuo feature pipeline → Tafkir classification
2. Ensemble (Tribuo XGBoost + Tafkir NN)
3. Tribuo text classification → Tafkir
4. Tribuo regression + Tafkir refinement

**Key Classes**:
- `TribuoFeaturePipelineTafkirClassifier`
- `EnsembleTribuoXGBoostTafkir`
- `TribuoTextPipelineTafkirClassifier`
- `TribuoRegressionTafkirRefinement`

**Run**:
```bash
jbang tribuo_integration.java
```

**Dependencies**:
- tribuo-all
- tafkir-sdk-nn

## 🔧 Common Utilities

### Tensor Conversion (Deeplearning4j)

```java
// DL4J → Tafkir
INDArray indArray = Nd4j.create(data, shape);
GradTensor tensor = indArrayToGradTensor(indArray);

// Tafkir → DL4J
GradTensor tensor = GradTensor.of(data, shape);
INDArray indArray = gradTensorToIndArray(tensor);
```

### Dataset Generation

All examples include demo dataset generators:

```java
// Classification data
MutableDataset<Label> dataset = DemoDataset.generateClassificationData(
    numSamples, numFeatures, random
);

// Regression data
MutableDataset<Regressor> dataset = DemoDataset.generateRegressionData(
    numSamples, numFeatures, random
);

// Text data
List<String> texts = DemoDataset.getSampleTexts();
List<String> labels = DemoDataset.getSampleLabels();
```

## 📊 Performance Tips

### First Run Performance

First execution downloads dependencies (1-2 minutes):
```bash
# Pre-download dependencies
jbang cache deps deeplearning4j_integration.java

# Subsequent runs are fast (1-2 seconds)
jbang deeplearning4j_integration.java
```

### Memory Configuration

Some libraries require significant memory:
```bash
# Increase heap size
JAVA_OPTS="-Xmx4g" jbang stanford_nlp_integration.java
```

### Model Caching

Cache loaded models for reuse:
```java
// Static model instance
static final Model CACHED_MODEL = loadModel();

// Pre-warm
static {
    CACHED_MODEL.forward(warmupInput);
}
```

## 🐛 Troubleshooting

### Dependency Download Failures

```bash
# Clear jbang cache
jbang cache clear

# Retry
jbang deeplearning4j_integration.java
```

### Native Library Errors

Ensure `--enable-native-access=ALL-UNNAMED` is in JAVA_OPTIONS (already included in examples).

### Memory Issues

```bash
# Increase memory
export JAVA_OPTS="-Xmx4g -Xms2g"
jbang smile_ml_integration.java
```

### Model Download Failures (OpenNLP)

```bash
# Manually download models
mkdir -p /tmp/opennlp-models
cd /tmp/opennlp-models
wget https://opennlp.sourceforge.net/models-1.5/en-sent.bin
# ... other models

# Re-run example
jbang opennlp_integration.java
```

## 📚 Additional Resources

### Library Documentation

- [Deeplearning4j](https://deeplearning4j.konduit.ai/)
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
- [Apache OpenNLP](https://opennlp.apache.org/)
- [Smile ML](https://haifengl.github.io/)
- [Oracle Tribuo](https://tribuo.org/)

### Tafkir SDK

- [Tafkir SDK README](../../../README.md)
- [JBANG Setup Guide](../JBANG_SETUP.md)
- [Examples Directory](../examples/)

### jbang

- [jbang Documentation](https://jbang.dev/documentation)
- [jbang Templates](https://github.com/jbangdev/jbang-templates)

## 🤝 Contributing

When adding new integration examples:

1. Follow naming convention: `{library}_integration.java`
2. Include comprehensive comments
3. Demonstrate 2-3 use cases
4. Include error handling
5. Add to README.md
6. Test with `jbang {file}.java`

## 📄 License

These examples are provided under the same license as the Tafkir SDK.

---

**Happy integrating! 🚀**



================================================
FILE: examples/docs/SETUP.md
================================================
# Tafkir Examples Setup

Quick setup guide for running Tafkir examples.

---

## Prerequisites

| Requirement | Version | How to Check |
|-------------|---------|--------------|
| **Java** | 25+ | `java -version` |
| **JBang** | 0.120+ | `jbang --version` |
| **Jupyter** (optional) | Latest | `jupyter --version` |
| **Maven** (optional) | 3.8+ | `mvn --version` |

---

## Option 1: JBang (Easiest)

No build tools needed. JBang handles everything:

```bash
# Install JBang
curl -Ls https://sh.jbang.dev | bash -s -

# Run an example
cd tafkir/examples/jbang/
jbang common/hello_tafkir.java
```

**Available Examples:**
```bash
# SDK examples
jbang sdk/unified_framework_demo.java --demo all
jbang sdk/graph_fusion_example.java

# Edge inference
jbang edge/image_classification.java

# Quantization
jbang quantizer/tafkir-quantizer-turboquant.java --mode demo
```

---

## Option 2: Jupyter Notebooks

Interactive Python-style experience for Java:

```bash
# Install Jupyter
pip install jupyter

# Install Java kernel
jbang app install --force jupyter-java@quarkusio

# Launch
cd tafkir/examples/jupyter/
jupyter notebook
```

**Available Notebooks:**
- `01-getting-started.ipynb` - SDK basics
- `06-llm-integration.ipynb` - LLM integration

---

## Option 3: Maven Project

For full IDE support:

```bash
# Build the examples
cd tafkir/framework/lib/tafkir-ml-examples/
mvn clean compile

# Run an example
mvn exec:java -Dexec.mainClass=tech.kayys.tafkir.train.examples.SimpleFFNExample
```

---

## GPU Acceleration

### CUDA (NVIDIA)

```bash
# Verify CUDA availability
nvidia-smi

# Run with CUDA
jbang --java 25 sdk/unified_framework_demo.java --device cuda
```

### Metal (Apple Silicon)

Works automatically on M1/M2/M3:

```bash
# Run with Metal
jbang --java 25 sdk/unified_framework_demo.java --device metal
```

---

## Troubleshooting

### "Could not find or load main class"

Ensure Java 25:

```bash
java -version
# Should show: openjdk version "25.x.x"
```

### "Dependency resolution failed"

Clear JBang cache:

```bash
jbang cache clear
jbang sdk/unified_framework_demo.java
```

### "Native memory access denied"

Add native access flags to JBang script header:

```java
//COMPILE_OPTIONS --enable-native-access=ALL-UNNAMED
//RUNTIME_OPTIONS --enable-native-access=ALL-UNNAMED
```

### Jupyter kernel not found

Install the Java kernel:

```bash
jbang app install --force jupyter-java@quarkusio
jupyter kernelspec list  # Should show 'java' or 'jjava'
```

---

## Next Steps

- **[JBang Setup Guide](/docs/setup/jbang-setup) - Detailed JBang configuration
- **[SDK Installation](/docs/setup/sdk-installation) - Maven/Gradle setup
- **[Examples Index](/examples/jbang/INDEX.md) - Browse all examples
- **[Framework Guide](/docs/framework/) - Learn the ML framework



================================================
FILE: examples/jbang/INDEX.md
================================================
# Tafkir JBang Examples

This folder is the runnable JBang surface for Tafkir demos and integration
playbooks. It tracks the Gradle-first module vision (`ml/*`, `trainer/*`,
`integration/*`) while keeping older SDK-named scripts for compatibility.

## Quick Start

1. Install JBang:
   - `brew install jbang/tap/jbang`
2. Prepare local artifacts from `tafkir/` project root:
   - `./run-install-local-macos.sh`
   - `./gradlew publishJbangTrainerExamplesToMavenLocal`
   - `./gradlew smokeJbangTrainerRuntimeProfileBudgetGate`
   - `./gradlew verifyJbangTrainerRuntimeProfileBudgetGateOutput`
   - `./gradlew smokeJbangTrainerQualityProfileCiGateEvidence`
   - `./gradlew verifyJbangTrainerQualityProfileCiGateEvidenceOutput`
   - `./gradlew smokeJbangTrainerExamples`
   - `./gradlew smokeJbangAgentBridgeExamples -Ptafkir.jbang.agentBridge.offline=false`
   - The Gradle smoke lane uses `build/jbang/.jbang` plus offline JBang
     resolution after the local artifacts are published. Add
     `-Ptafkir.jbang.trainer.offline=false` only to refresh remote JBang
     dependencies intentionally.
3. Run examples from `tafkir/examples/jbang`:
   - `jbang trainer/trainer_runtime_bootstrap.java`
   - `jbang trainer/trainer_runtime_profile_budget_gate.java`
   - `jbang trainer/trainer_quality_profile_ci_gate_evidence.java`
   - `jbang sdk/tafkir-quickstart.java`
   - `jbang sdk/tafkir-sdk-core-example.java`
   - `jbang sdk/tafkir-sdk-train-example.java 2`
   - `jbang sdk/tafkir-sdk-vision-example.java`
   - `jbang sdk/tafkir-sdk-export-example.java`
   - `jbang sdk/tafkir-sdk-augment-example.java`
   - `jbang run --no-integrations integration/wayang_tafkir_serving_bridge.java --mock`

## Recommended Paths

### Trainer Runtime (Canonical)

- `trainer/trainer_runtime_bootstrap.java`
  - Uses `tafkir-trainer` + `tafkir-trainer-api`
  - Prints runtime mode (`legacy-bridge` vs `canonical-fallback`)
  - Shows listener lifecycle and training summary
- `trainer/trainer_runtime_profile_budget_gate.java`
  - Uses `tafkir-ml-api`
  - Builds a tiny canonical runtime-profile report, evaluates strict trainer
    runtime budgets, writes JSON/Markdown/JUnit evidence, and verifies the
    persisted artifact bundle
  - Supports `--out`, named policies, threshold overrides, and `--fail-on-gate`
- `trainer/trainer_quality_profile_ci_gate_evidence.java`
  - Uses `tafkir-ml-api` to train tiny baseline/candidate reports
  - Runs a quality-profile CI gate, writes the manifest, writes
    JSON/Markdown/JUnit verification evidence, and verifies the persisted
    evidence bundle
  - Supports `--out`, `--device`, `--profile`, epoch controls, and
    `--fail-on-gate`
- `trainer/trainer_diffusion_opd_ddim.java`
  - Uses `tafkir-ml-api` + `tafkir-ml-diffusion-opd` + `tafkir-diffusion`
  - Demonstrates Java-first DiffusionOPD wiring with DDIM-style scheduler flow
  - Shows task prompts, teacher adapters, and `Tafkir.DL.diffusionOpdTrainer()`
- `trainer/trainer_diffusion_opd_stable_diffusion_metal_bridge.java`
  - Uses `tafkir-ml-diffusion-opd` + `tafkir-backend-metal` +
    `tafkir-runner-stable-diffusion`
  - Demonstrates the Metal-aware stable-diffusion bridge round-trip path
  - Shows accelerator metadata and scheduler alignment for OPD integration
- `trainer/trainer_diffusion_opd_stable_diffusion_native_bridge.java`
  - Uses `tafkir-ml-api` + `tafkir-ml-diffusion-opd` +
    `tafkir-runner-stable-diffusion`
  - Demonstrates end-to-end OPD wiring with stable-diffusion-native UNet and
    executable PNDM scheduler bridge surfaces together
  - Shows task routing, adaptive stage weighting, trainer artifact output, and
    real CLIP prompt conditioning when `text_encoder/` and `tokenizer/` are
    present
  - Supports task-specific CLIP fixture overrides for `ocr` and `aesthetics`
- `trainer/trainer_diffusion_opd_report_inspector.java`
  - Uses `tafkir-ml-diffusion-opd`
  - Loads `diffusion-opd-report.json` and prints either the full report or a
    selected section such as `overview`, `conditioning`, `teachers`, or
    `roundHistory`, in `text`, pretty-printed `json`, `table`, or `csv`
  - Supports grouped rollups like `taskSummaries` and `teacherSummaries`
  - Supports ranked grouped rollups like `taskSummaries:sort=-meanLoss:top=5`
  - Supports grouped pair rollups like
    `taskTeacherSummaries:sort=-meanLoss:top=10`
  - Supports export column projection like `value,count,meanLoss`
  - Supports optional export-to-file output paths for `text`, `json`, `table`,
    and `csv`; `_` remains available as a no-projection placeholder when
    needed, but most `json`/`csv` cases can now use filename inference
  - Can infer export format from output filenames like `.json`, `.csv`,
    `.table`, `.md`, `.txt`, or `.log`
  - Supports directory-oriented bundle export with `bundle=standard` and
    `bundle=rollups`
  - Supports custom bundle definitions like
    `bundle=custom:overview,taskSummaries,taskTeacherSummaries`
  - Custom bundles support inline aliases like
    `bundle=custom:overview@run-overview,taskTeacherSummaries@teacher-pairs`
  - Custom bundles support per-entry columns like
    `bundle=custom:taskSummaries@tasks#compact,taskTeacherSummaries@pairs#compare`
  - Custom bundles support per-entry formats like
    `bundle=custom:overview@run-overview!json,taskTeacherSummaries@teacher-pairs!csv`
  - Bundle directories include a `manifest.json` index with timestamps and
    per-file section metadata for downstream tooling
  - Bundle directories can be inspected directly with `manifest`,
    `bundleSummary`, `bundleHealth`, `files`, `filesSummary:by=section`,
    `filesSummary:by=format:sort=-count`,
    `filesSummary:by=section:sort=-count:top=3`,
    `files:format=csv`, `files:section=taskTeacherSummaries`, or
    `file:<name>` and `loadfile:...` views, with
    `sort=name|-name|section|count`, `top=<n>`, `pick=last`, or `index=<n>`
    selectors for multi-match cases
  - `bundleSummary` supports `focus=sections|formats|all` and `top=<n>` for
    compact rollup views, plus `dominant=true` for headline-only output
  - `bundleSummary` also reports missing bundle files referenced by the
    manifest
  - `bundleHealth` gives a compact pass/fail integrity view and groups
    missing files by section and format, with normalized `status`,
    `healthScore`, `alertLevel`, stable `issueCodes`, compact `healthBadge`
    with UI-ready `variant`, `label`, `token`, `tooltip`, `checkStatus`,
    `primaryCheckCode`, and `primaryCheckSeverity`,
    and human-facing `summaryMessage` / `recommendedAction`, plus structured named checks,
    normalized `checkSummary` including pass/failure rates, local status/alert fields, local guidance strings, local issue codes, local primary issue code, a nested badge with its own primary issue code, primary check name/message/severity, summary/action guidance, token/tooltip/check-status support, and direct action hints, `allPassed`, fast critical/warning/info failure flags, failing severity count breakdowns, primary failing check identity/severity/message, compact failing names/codes, compact failing-check summaries, and aggregate
    check counters
  - `bundleHealth` supports `focus=files|sections|formats|all` and `top=<n>`
    for compact diagnosis slices
  - `table`/`csv` bundles use section-aware default projections like
    `compact` for grouped summaries and `compare` for pair rollups
  - Supports alias columns like `loss`, `latestRound`, `latestLoss`,
    `top1Round`, `top1Loss`, `first1Round`, and `first1Loss`
  - Supports projection presets like `minimal`, `compact`, `latest`,
    `leaderboard`, `details`, and `compare`
  - `details` and `compare` can flatten both top-loss and first-round
    representative rows into export-friendly columns
  - Presets adapt to row shape, using `pair` for pair rollups and `value` for
    grouped single-dimension rollups
  - Supports compact presets like `taskSummary=ocr` and
    `teacherSummary=ocr-early`
  - Supports pair presets like `taskTeacherSummary=ocr,ocr-early`
    and `taskStageSummary=ocr,early`
  - Supports focused selectors like `roundHistory:last` and
    `roundHistory:task=ocr`
  - Supports compact queries like `roundHistory:task=ocr:last` and
    `roundHistory:task=ocr:meanLoss`
  - Supports scalar aliases and summary rollups like
    `roundHistory:teacher=ocr-early:lastLoss` and
    `roundHistory:task=ocr:summary`
  - Supports ranked queries like
    `roundHistory:task=ocr:sort=-averageLoss:top=3`
- `trainer/trainer_byte_latent_history_inspector.java`
  - Uses `tafkir-ml-byte-latent`
  - Accepts either a byte-latent checkpoint directory or direct
    `byte-latent-history.csv` path
  - Supports `status` as the shortest checkpoint-side status selector
  - Supports `ci` as the shortest compact CI-style checkpoint status selector
  - Supports `status:ci` as the explicit CI-style checkpoint status selector
  - Supports compact `history:` selectors like `history:summary`,
    `history:lastLoss`, `history:epoch=2`, and
    `history:sort=-trainLoss:top=3`
  - Supports `text`, pretty-printed `json`, and spreadsheet-ready `csv`
    output, with optional export-to-file path as a fourth argument
- `trainer/trainer_byte_latent_demo.java`
  - Uses `tafkir-ml-byte-latent`
  - Trains a small byte-latent demo run from UTF-8 text lines
  - Writes `byte-latent-summary.json`, `byte-latent-history.csv`,
    `byte-latent-report.json`, and the checkpoint manifest
  - Prints ready-to-run follow-up inspector commands for the generated
    checkpoint directory
- `trainer/trainer_byte_latent_resume_demo.java`
  - Uses `tafkir-ml-byte-latent`
  - Seeds a one-epoch byte-latent checkpoint and then resumes it to a longer
    target epoch count
  - Shows resumed history growth, loaded resume metadata, and continued
    report/checkpoint inspection commands
- `trainer/trainer_byte_latent_infer_demo.java`
  - Uses `tafkir-ml-byte-latent`
  - Runs a tiny reference byte-latent prompt continuation flow
  - Prints prompt token ids, `predictNextToken(...)`, generated token ids, and
    combined byte/text output for a short deterministic continuation
- `trainer/trainer_byte_latent_train_infer_demo.java`
  - Uses `tafkir-ml-byte-latent`
  - Runs a tiny trainer session and then immediately performs prompt continuation
  - Prints training summary, checkpoint paths, next-token prediction, and short
    deterministic generation from the same model family/spec
  - Writes `train-infer-report.json` with both training and inference sections
- `trainer/trainer_byte_latent_train_infer_inspector.java`
  - Uses Jackson only and reads `train-infer-report.json`
  - Supports `overview`, `training`, `inference`, `nextToken`,
    `generatedText`, `combinedText`, `status`, and `ci`
- `trainer/trainer_byte_latent_workflow_inspector.java`
  - Uses Jackson only and reads `byte_latent_workflow_manifest.json`
  - Supports `overview`, `runs`, `workflowBundleDir`,
    `workflowBundleStatus`, `run:fresh`, `historyRows`,
    `delta:historyRows`, and `delta:latestTrainLoss`
  - Supports `bundleOverview` and
    `bundleOverview:focus=health|files|all` plus `summary=short` for compact
    bundle summaries
  - Supports `bundleOverview:ci` as the shorthand alias for the recommended
    CI-style overview check
  - Supports `status` as the shortest neutral bundle check selector, with
    `bundleStatus` kept as the explicit form
  - Supports `bundleHealth:ci` and `bundleHealth:summary=short` for single-line
    direct health checks
  - Can jump directly from a workflow manifest into a materialized bundle with
    `bundleManifest`, `bundleFiles`, `bundleSummary`, `bundleHealth`, and
    `bundleLoadfile:section=<section>`
  - Supports `bundle=standard` to export overview, runs, and delta views into
    one output directory with a small bundle manifest
  - Can inspect exported bundle directories with `bundleSummary`,
    `bundleHealth`, `files`, `files:section=<section>`,
    `files:name=<name>`, `file:<name>`, and `loadfile:section=<section>`
  - Supports `files:...:sort=name|section|format` for compact ordered listings
  - Supports `loadfile:section=<section>:pick=first|last` and `index=<n>`
  - Supports `text`, pretty-printed `json`, and spreadsheet-ready `csv`
- `trainer/run_byte_latent_smoke.sh`
  - Runs the byte-latent demo plus key inspector checks in one command
  - Verifies the expected checkpoint artifacts exist
  - Prints follow-up commands for resumed-run inspection
- `trainer/run_byte_latent_workflow.sh`
  - Unifies fresh-run, resumed-run, and full smoke flows behind one entrypoint
  - Supports `fresh`, `resume`, and `full` modes plus optional checkpoint-root
  - Prints a compact final workflow summary with checkpoint paths, history row
    counts, and inspector-ready commands
  - Writes a machine-readable workflow manifest JSON for CI or follow-up tools
  - Records `workflowBundleDir` and `workflowBundleStatus` in that manifest
  - Materializes a workflow bundle automatically in `full` mode so the printed
    `loadfile:` examples point at a ready directory
  - Prints ready-to-run workflow-manifest inspector commands at the end
  - Delegates to the existing smoke helpers so older commands keep working
- `trainer/run_byte_latent_resume_smoke.sh`
  - Runs the byte-latent resume demo plus key inspector checks in one command
  - Verifies the resumed checkpoint artifacts and persisted history growth
  - Prints follow-up commands for continued checkpoint inspection

### SDK Compatibility Scripts (Runnable Today)

- `sdk/tafkir-quickstart.java`
- `sdk/tafkir-sdk-core-example.java`
- `sdk/tafkir-sdk-train-example.java`
- `sdk/tafkir-sdk-vision-example.java`
- `sdk/tafkir-sdk-export-example.java`
- `sdk/tafkir-sdk-augment-example.java`

These are compatibility entrypoints with legacy file names, but they now run
against the current local artifact set.

### Agent Serving Bridge

- `integration/wayang_tafkir_serving_bridge.java`
  - Uses the lightweight `tafkir-sdk-agent` module directly
  - The full `tafkir-sdk-remote` client still exposes the same agent client
    through `client.agent()`
  - Demonstrates Tafkir as the serving/inference engine for Wayang-Tafkir
  - Shows capability and contract discovery, MCP tool schema discovery,
    request validation, embeddings, caller-owned RAG retrieval, streamed chat
    deltas, streamed tool-call previews, tool-call handoff, and Wayang-owned
    memory
  - Runs with `--mock` by default so the flow can be inspected without a live
    server; use `jbang run --no-integrations ...` because this plain Java
    bridge does not need JBang's Quarkus post-build integration
  - CI/local smoke: `./gradlew smokeJbangAgentBridgeExamples
    -Ptafkir.jbang.agentBridge.offline=false`
  - Smoke output is written to
    `build/jbang/integration/wayang-tafkir-serving-bridge.out`
  - Use `--live --base-url http://localhost:8080 --api-key community` against
    a running Tafkir endpoint after publishing
    `:sdk:tafkir-sdk-agent:publishToMavenLocal`

### Experimental v0.2/v0.3 Scripts

- `sdk/tensor_operations_v02.java`
- `sdk/vision_transforms_v02.java`
- `sdk/tokenization_v02.java`
- `sdk/mnist_training_v02.java`
- `sdk/pytorch_comparison_v02.java`
- `sdk/unified_framework_demo.java`
- `sdk/graph_fusion_example.java`

These are useful references, but dependency coordinates vary and may require
extra local publishing beyond the quickstart path.

### Inference and Modality Demos

- `nlp/*`
- `multimodal/*`
- `edge/*`
- `quantizer/*`
- `integration/*`

## Directory Map

| Folder | Focus | Notes |
|---|---|---|
| `trainer/` | Canonical trainer runtime | Aligned with `:trainer:*` modules |
| `sdk/` | ML framework and compatibility examples | Mixed canonical + legacy naming |
| `nlp/` | NLP pipelines and chat demos | Includes SIMD and GGUF flows |
| `multimodal/` | Vision/audio/text demos | Mixed quality, evolving APIs |
| `edge/` | LiteRT and edge paths | Device-oriented scenarios |
| `quantizer/` | Quantization experiments | AWQ/GPTQ/TurboQuant samples |
| `integration/` | 3rd-party integrations and agent bridges | Wayang-Tafkir, DL4J, Smile, Tribuo, OpenNLP |
| `common/` | Baseline utilities | Legacy starter scripts |

## Compatibility Notes

- Scripts named `tafkir-sdk-*` are compatibility examples, not canonical naming.
- Some older examples depend on `${user.home}/.tafkir/jbang/libs/*` local jars.
- Prefer scripts that use `//REPOS local,mavencentral` plus
  targeted `publishToMavenLocal` tasks.



================================================
FILE: examples/jbang/QUICKSTART.txt
================================================
===================================================================
 TAFKIR JBANG QUICKSTART (GRADLE-FIRST, 2026)
===================================================================

1) One-time setup
-------------------------------------------------------------------
Install JBang:
  brew install jbang/tap/jbang

From tafkir project root, prepare local artifacts:
  ./run-install-local-macos.sh

2) Move to examples
-------------------------------------------------------------------
cd tafkir/examples/jbang

3) Run recommended entrypoints
-------------------------------------------------------------------
Canonical trainer runtime:
  jbang trainer/trainer_runtime_bootstrap.java

Byte-latent smoke flow:
  ./trainer/run_byte_latent_workflow.sh
  ./trainer/run_byte_latent_workflow.sh fresh
  ./trainer/run_byte_latent_workflow.sh resume
  ./trainer/run_byte_latent_workflow.sh full /tmp/tafkir-byte-latent
  # prints a compact final fresh/resume checkpoint summary in full mode
  # also writes a small workflow manifest JSON for CI or follow-up scripts
  # and materializes a workflow bundle automatically in full mode
  # and prints ready-to-run workflow-inspector commands for that manifest
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json overview
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json workflowBundleDir
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview:focus=health
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview:focus=files
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview:focus=health:summary=short
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json status
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json health
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleStatus
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview:ci
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleHealth:ci
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleHealth:summary=short
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:summary
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:ci
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:ci:mode=fresh
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json inspectfields
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json inspectfields:summary
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json inspectfields:ci
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands:all
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands:summary
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands:ci
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands:fresh
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runfields
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runfields:summary
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runfields:ci
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runfields:fresh
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:scope=workflow
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:mode=fresh
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview json
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleManifest json
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleFiles json
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleLoadfile:section=runs json
  jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundle=standard json /tmp/byte-latent-workflow-bundle
  jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle bundleSummary json
  jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle bundleHealth json
  jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files:section=runs json
  jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files:name=overview.json json
  jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle filesSummary:by=section:sort=-count:top=3 json
  jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle file:overview.json json
  jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle loadfile:section=runs json
  jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle loadfile:section=runs:index=0 json
  jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo status
  jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo health
  jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo ci
  jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo status:ci
  ./trainer/run_byte_latent_smoke.sh
  ./trainer/run_byte_latent_smoke.sh /tmp/tafkir-byte-latent-smoke

Byte-latent resume smoke flow:
  ./trainer/run_byte_latent_resume_smoke.sh
  ./trainer/run_byte_latent_resume_smoke.sh /tmp/tafkir-byte-latent-resume-smoke

Byte-latent inference demo:
  jbang trainer/trainer_byte_latent_infer_demo.java
  jbang trainer/trainer_byte_latent_infer_demo.java "hi"
  jbang trainer/trainer_byte_latent_infer_demo.java "tafkir" 6

Byte-latent train+infer demo:
  jbang trainer/trainer_byte_latent_train_infer_demo.java
  jbang trainer/trainer_byte_latent_train_infer_demo.java /tmp/byte-latent-train-infer "hi" 4
  # writes /tmp/byte-latent-train-infer/train-infer-report.json
  jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer overview
  jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer training
  jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer nextToken
  jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer combinedText
  jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer ci

SDK compatibility lane (verified runnable):
  jbang sdk/tafkir-quickstart.java
  jbang sdk/tafkir-sdk-core-example.java
  jbang sdk/tafkir-sdk-train-example.java 2
  jbang sdk/tafkir-sdk-vision-example.java
  jbang sdk/tafkir-sdk-export-example.java
  jbang sdk/tafkir-sdk-augment-example.java

Experimental lane (may require extra local publishing):
  jbang sdk/tensor_operations_v02.java
  jbang sdk/vision_transforms_v02.java
  jbang sdk/tokenization_v02.java

===================================================================
 NOTES
===================================================================

- Older scripts named tafkir-sdk-* are kept as compatibility examples.
- Some legacy samples still depend on local jars under:
  ${user.home}/.tafkir/jbang/libs
- If dependency resolution fails, re-run targeted publish:
  ./gradlew :trainer:tafkir-trainer-api:publishToMavenLocal :trainer:tafkir-trainer:publishToMavenLocal

===================================================================
 TROUBLESHOOTING
===================================================================

JBang not found:
  export PATH="$PATH:$HOME/.jbang/bin"

Force refresh JBang cache:
  jbang --fresh trainer/trainer_runtime_bootstrap.java

Need higher heap for large demos:
  jbang --java-options="-Xmx4g" sdk/mnist_training_v02.java



================================================
FILE: examples/jbang/common/batch_process.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import java.util.*;

public class batch_process {
    public static void main(String[] args) {
        Sequential model = new Sequential(
            new Linear(10, 5),
            new ReLU(),
            new Linear(5, 1)
        );
        
        // Batch data
        List<String> files = Arrays.asList("file1.txt", "file2.txt", "file3.txt");
        
        for (String file : files) {
            System.out.println("Processing " + file);
            // Process...
        }
        
        System.out.println("Batch processing complete!");
    }
}



================================================
FILE: examples/jbang/common/CompleteMlExample.java
================================================
package tech.kayys.tafkir.ml.example;

import tech.kayys.tafkir.ml.base.*;
import tech.kayys.tafkir.ml.pipeline.*;
import tech.kayys.tafkir.ml.ensemble.*;
import tech.kayys.tafkir.ml.model_selection.*;
import tech.kayys.tafkir.ml.metrics.*;

/**
 * Complete example showing all components working together.
 */
public class CompleteMlExample {
    
    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║           Complete Tafkir ML Framework Demo                  ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝\n");
        
        try {
            // Load sample data
            float[][] X = loadSampleData();
            int[] y = loadSampleLabels();
            
            System.out.println("📊 Data loaded:");
            System.out.printf("   Samples: %d, Features: %d, Classes: %d\n", 
                X.length, X[0].length, Arrays.stream(y).max().getAsInt() + 1);
            
            // 1. Create preprocessing pipeline
            System.out.println("\n🔧 Building preprocessing pipeline...");
            Pipeline preprocessor = new Pipeline(
                new BaseTransformer[]{
                    new StandardScaler(),
                    new PCA(2)
                },
                null
            );
            
            float[][] XProcessed = preprocessor.fitTransform(X);
            System.out.printf("   Features after PCA: %d\n", XProcessed[0].length);
            
            // 2. Train multiple models
            System.out.println("\n🤖 Training models...");
            
            // Random Forest
            RandomForestClassifier rf = new RandomForestClassifier();
            rf.fit(XProcessed, y);
            double rfAcc = rf.score(XProcessed, y);
            System.out.printf("   Random Forest Accuracy: %.4f\n", rfAcc);
            
            // SVM
            SVC svm = new SVC();
            svm.fit(XProcessed, y);
            double svmAcc = svm.score(XProcessed, y);
            System.out.printf("   SVM Accuracy: %.4f\n", svmAcc);
            
            // Gradient Boosting
            GradientBoostingClassifier gb = new GradientBoostingClassifier();
            gb.fit(XProcessed, y);
            double gbAcc = gb.score(XProcessed, y);
            System.out.printf("   Gradient Boosting Accuracy: %.4f\n", gbAcc);
            
            // 3. Cross-validation
            System.out.println("\n📈 Cross-validation results (5-fold):");
            
            KFold kfold = new KFold(5, true, 42);
            List<Fold> folds = kfold.split(XProcessed.length);
            
            double cvSum = 0;
            for (int i = 0; i < folds.size(); i++) {
                Fold fold = folds.get(i);
                
                // Extract fold data
                float[][] XTrain = new float[fold.trainIndices.length][];
                int[] yTrain = new int[fold.trainIndices.length];
                float[][] XVal = new float[fold.valIndices.length][];
                int[] yVal = new int[fold.valIndices.length];
                
                for (int j = 0; j < fold.trainIndices.length; j++) {
                    int idx = fold.trainIndices[j];
                    XTrain[j] = XProcessed[idx];
                    yTrain[j] = y[idx];
                }
                for (int j = 0; j < fold.valIndices.length; j++) {
                    int idx = fold.valIndices[j];
                    XVal[j] = XProcessed[idx];
                    yVal[j] = y[idx];
                }
                
                // Train and evaluate
                RandomForestClassifier foldRf = new RandomForestClassifier();
                foldRf.fit(XTrain, yTrain);
                double acc = Metrics.accuracy(yVal, foldRf.predict(XVal));
                cvSum += acc;
                System.out.printf("   Fold %d: %.4f\n", i + 1, acc);
            }
            System.out.printf("   Mean CV Accuracy: %.4f\n", cvSum / folds.size());
            
            // 4. Model selection with grid search
            System.out.println("\n🔍 Grid search for best parameters...");
            Map<String, Object[]> paramGrid = new LinkedHashMap<>();
            paramGrid.put("nEstimators", new Object[]{50, 100});
            paramGrid.put("maxDepth", new Object[]{5, 10});
            paramGrid.put("minSamplesSplit", new Object[]{2, 5});
            
            // 5. Ensemble voting
            System.out.println("\n🎯 Voting ensemble:");
            List<BaseEstimator> estimators = List.of(rf, svm, gb);
            VotingClassifier voting = new VotingClassifier(estimators, "hard", new double[]{1, 1, 1});
            double votingAcc = voting.score(XProcessed, y);
            System.out.printf("   Voting Classifier Accuracy: %.4f\n", votingAcc);
            
            // 6. Model persistence
            System.out.println("\n💾 Saving best model...");
            ModelPersistence.save(rf, "random_forest_model.tafkir");
            System.out.println("   Model saved to random_forest_model.tafkir");
            
            // 7. Load and predict
            System.out.println("\n📂 Loading saved model...");
            RandomForestClassifier loadedRf = ModelPersistence.load("random_forest_model.tafkir");
            int[] predictions = loadedRf.predict(XProcessed);
            double loadedAcc = Metrics.accuracy(y, predictions);
            System.out.printf("   Loaded model accuracy: %.4f\n", loadedAcc);
            
            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║                    ✓ COMPLETE!                               ║");
            System.out.println("║    All ML components working together successfully          ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static float[][] loadSampleData() {
        // Generate synthetic 2D data (3 blobs for clustering/classification)
        float[][] data = new float[300][2];
        Random rng = new Random(42);
        
        for (int i = 0; i < 300; i++) {
            int cluster = i / 100;
            double angle = rng.nextDouble() * Math.PI * 2;
            double radius = rng.nextDouble() * 2;
            
            double x = Math.cos(angle) * radius + (cluster == 0 ? 0 : cluster == 1 ? 5 : -5);
            double y = Math.sin(angle) * radius + (cluster == 0 ? 0 : cluster == 1 ? 5 : -5);
            
            data[i][0] = (float) (x + rng.nextGaussian() * 0.3);
            data[i][1] = (float) (y + rng.nextGaussian() * 0.3);
        }
        
        return data;
    }
    
    private static int[] loadSampleLabels() {
        int[] labels = new int[300];
        for (int i = 0; i < 300; i++) {
            labels[i] = i / 100;
        }
        return labels;
    }
}


================================================
FILE: examples/jbang/common/custom_module_demo.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import tech.kayys.tafkir.ml.autograd.GradTensor;

/**
 * Advanced example: Creating a custom Module structure.
 * This demonstrates how to build modular, reusable architectures beyond Sequential.
 */
class ResidualBlock extends Module {
    private final Linear fc1;
    private final Linear fc2;
    private final ReLU relu = new ReLU();

    public ResidualBlock(int dim) {
        this.fc1 = register("fc1", new Linear(dim, dim));
        this.fc2 = register("fc2", new Linear(dim, dim));
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // x = ReLU(fc2(ReLU(fc1(input))) + input)
        GradTensor out = relu.forward(fc1.forward(input));
        out = fc2.forward(out);
        return relu.forward(out.add(input)); // Residual connection
    }
}

public class custom_module_demo {
    public static void main(String[] args) {
        System.out.println("=== Custom Module Architecture ===");

        // Building a model using our custom block
        Sequential model = new Sequential(
            new Linear(64, 64),
            new ResidualBlock(64),
            new ResidualBlock(64),
            new Linear(64, 10)
        );

        System.out.println("Model with residual connections created:");
        System.out.println(model);
        
        GradTensor input = GradTensor.randn(1, 64);
        GradTensor output = model.forward(input);
        
        System.out.println("Output shape: " + java.util.Arrays.toString(output.shape()));
    }
}



================================================
FILE: examples/jbang/common/error_handling.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;

public class error_handling {
    public static void main(String[] args) {
        try {
            Sequential model = new Sequential(
                new Linear(784, 128),
                new ReLU(),
                new Linear(128, 10)
            );
            
            System.out.println("Model created successfully!");
            
        } catch (IllegalArgumentException e) {
            System.err.println("Invalid model configuration: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}



================================================
FILE: examples/jbang/common/hello_tafkir.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;

public class hello_tafkir {
    public static void main(String[] args) {
        System.out.println("Hello from Tafkir!");
        
        // Create a simple model
        Sequential model = new Sequential(
            new Linear(10, 5),
            new ReLU(),
            new Linear(5, 1)
        );
        
        System.out.println("Model created successfully!");
    }
}



================================================
FILE: examples/jbang/common/mnist_style_setup.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import tech.kayys.tafkir.ml.nn.loss.*;
import tech.kayys.tafkir.ml.nn.optim.*;
import tech.kayys.tafkir.ml.autograd.GradTensor;

/**
 * Simulates a complex Digit Classification setup (like MNIST).
 * Demonstrates deeper networks, parameter counting, and training/eval modes.
 */
public class mnist_style_setup {
    public static void main(String[] args) {
        int inputSize = 28 * 28; // 784
        int hiddenSize = 256;
        int numClasses = 10;

        // 1. Advanced Model Architecture
        Sequential model = new Sequential(
            new Linear(inputSize, hiddenSize),
            new ReLU(),
            // new Dropout(0.2), // If supported
            new Linear(hiddenSize, hiddenSize / 2),
            new ReLU(),
            new Linear(hiddenSize / 2, numClasses)
        );

        System.out.println("--- Model Summary ---");
        System.out.println(model);
        System.out.println("Total Parameters: " + model.parameterCountFormatted());

        // 2. Training Components
        CrossEntropyLoss criterion = new CrossEntropyLoss();
        Adam optimizer = new Adam(model.parameters(), 0.001f);

        // 3. Batch simulation
        int batchSize = 32;
        GradTensor fakeImages = GradTensor.randn(batchSize, inputSize);
        GradTensor fakeLabels = GradTensor.zeros(batchSize, numClasses); // One-hot labels

        System.out.println("\n--- Step Execution ---");
        
        // Train mode
        model.train();
        GradTensor output = model.forward(fakeImages);
        GradTensor loss = criterion.compute(output, fakeLabels);
        
        System.out.println("Forward pass batch output shape: " + java.util.Arrays.toString(output.shape()));
        System.out.println("Current batch loss: " + loss.item());

        // Eval mode
        model.eval();
        GradTensor valOutput = model.forward(fakeImages);
        System.out.println("Evaluation mode predictions complete.");
    }
}



================================================
FILE: examples/jbang/common/model_persistence.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import java.nio.file.Paths;

/**
 * Demonstrates Model Persistence: Saving and Loading model weights.
 */
public class model_persistence {
    public static void main(String[] args) throws Exception {
        String modelPath = "xor_model.bin";
        
        System.out.println("Creating a new model...");
        Sequential model = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );

        // In a real scenario, you would train the model here...
        
        System.out.println("Saving model to " + modelPath + "...");
        model.save(Paths.get(modelPath));
        System.out.println("Model saved successfully!");

        System.out.println("\nCreating a fresh model instance...");
        Sequential newModel = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );

        System.out.println("Loading weights into the fresh model...");
        newModel.load(Paths.get(modelPath));
        System.out.println("Model weights loaded successfully!");
        
        // Clean up
        Paths.get(modelPath).toFile().deleteOnExit();
    }
}



================================================
FILE: examples/jbang/common/model_persistence_safetensor.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import tech.kayys.tafkir.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.tafkir.ml.nn.optim.SGD;
import tech.kayys.tafkir.ml.autograd.GradTensor;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * End-to-End Example: Training, Saving, Loading, and Evaluating with Safetensors.
 * 
 * This script demonstrates the full lifecycle of a Tafkir model:
 * 1. Define a non-linear XOR model.
 * 2. Train it for a few epochs.
 * 3. Save the trained weights to the universal Safetensor format.
 * 4. Create a fresh, random instance of the same model architecture.
 * 5. Load the trained weights via loadSafetensors().
 * 6. Verify that the reloaded model's predictions match the original.
 */
public class model_persistence_safetensor {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Tafkir End-to-End Persistence (Safetensor) ===");

        // --- 1. SETUP DATA & MODEL ---
        GradTensor x = GradTensor.of(new float[]{0,0, 0,1, 1,0, 1,1}, 4, 2);
        GradTensor y = GradTensor.of(new float[]{0, 1, 1, 0}, 4, 1);

        Sequential model = new Sequential(
            new Linear(2, 8),
            new ReLU(),
            new Linear(8, 1)
        );

        // --- 2. QUICK TRAINING ---
        System.out.println("\n[1/4] Training the original model...");
        long startTrain = System.nanoTime();
        BCEWithLogitsLoss criterion = new BCEWithLogitsLoss();
        SGD optimizer = new SGD(model.parameters(), 0.5f);

        for (int i = 1; i <= 200; i++) {
            GradTensor output = model.forward(x);
            GradTensor loss = criterion.compute(output, y);
            model.zeroGrad();
            loss.backward();
            optimizer.step();
            if (i % 50 == 0) System.out.printf("Epoch %d, Loss: %.4f\n", i, loss.item());
        }
        long trainTime = (System.nanoTime() - startTrain) / 1_000_000;
        System.out.printf("Training complete in %d ms.\n", trainTime);

        // Capture original prediction for later comparison
        float originalPred = model.forward(GradTensor.of(new float[]{1, 0}, 1, 2)).item();
        System.out.printf("Original model prediction for [1,0]: %.4f\n", originalPred);

        // --- 3. SAVE TO SAFETENSOR ---
        String filename = "trained_xor.safetensors";
        System.out.println("\n[2/4] Saving model to " + filename + "...");
        long startSave = System.nanoTime();
        model.saveSafetensors(filename);
        long saveTime = (System.nanoTime() - startSave) / 1_000_000;
        System.out.printf("Save complete in %d ms.\n", saveTime);

        // --- 4. RELOAD INTO FRESH MODEL ---
        System.out.println("\n[3/4] Creating a fresh model and loading weights...");
        Sequential freshModel = new Sequential(
            new Linear(2, 8),
            new ReLU(),
            new Linear(8, 1)
        );

        float randomPred = freshModel.forward(GradTensor.of(new float[]{1, 0}, 1, 2)).item();
        System.out.printf("Fresh (random) model prediction for [1,0]: %.4f\n", randomPred);

        System.out.println("Loading weights from Safetensor file...");
        long startLoad = System.nanoTime();
        freshModel.loadSafetensors(filename);
        long loadTime = (System.nanoTime() - startLoad) / 1_000_000;
        System.out.printf("Load complete in %d ms.\n", loadTime);

        // --- 5. EVALUATE & VERIFY ---
        System.out.println("\n[4/4] Evaluating reloaded model...");
        float reloadedPred = freshModel.forward(GradTensor.of(new float[]{1, 0}, 1, 2)).item();
        System.out.printf("Reloaded model prediction for [1,0]: %.4f\n", reloadedPred);

        if (Math.abs(originalPred - reloadedPred) < 1e-6) {
            System.out.println("\n✅ SUCCESS: Reloaded model matches original model perfectly!");
        } else {
            System.out.println("\n❌ FAILURE: Predictive discrepancy detected.");
        }
        
        // Clean up
        java.nio.file.Files.deleteIfExists(Path.of(filename));
    }
}



================================================
FILE: examples/jbang/common/my_script.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import tech.kayys.tafkir.ml.autograd.GradTensor;

public class my_script {
    public static void main(String[] args) {
        System.out.println("=== Tafkir Neural Network ===");
        
        // TODO: Add your code here
        
        Sequential model = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );
        
        System.out.println("Model ready for training!");
    }
}



================================================
FILE: examples/jbang/common/PickleExample.java
================================================
package tech.kayys.tafkir.ml.example;

import tech.kayys.tafkir.ml.base.*;
import tech.kayys.tafkir.ml.ensemble.*;
import tech.kayys.tafkir.ml.serialize.*;

/**
 * Complete example demonstrating pickle serialization.
 */
public class PickleExample {
    
    public static void main(String[] args) {
        try {
            System.out.println("╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║           Pickle Serialization Example                       ║");
            System.out.println("║      Save and Load Tafkir Models with Python Compatibility   ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝\n");
            
            // 1. Train a model
            System.out.println("1. Training Random Forest model...");
            RandomForestClassifier rf = new RandomForestClassifier();
            
            // Load sample data
            float[][] X = generateSampleData(1000);
            int[] y = generateSampleLabels(1000);
            
            rf.fit(X, y);
            double accuracy = rf.score(X, y);
            System.out.printf("   Training accuracy: %.4f\n", accuracy);
            
            // 2. Save as pickle
            System.out.println("\n2. Saving model as pickle...");
            PickleSerializer.toPickle(rf, "random_forest_model.pkl");
            System.out.println("   Saved: random_forest_model.pkl");
            
            // 3. Save as compressed pickle
            System.out.println("\n3. Saving compressed pickle...");
            PickleSerializer.toPickleGZ(rf, "random_forest_model.pkl.gz");
            System.out.println("   Saved: random_forest_model.pkl.gz");
            
            // 4. Load from pickle
            System.out.println("\n4. Loading model from pickle...");
            RandomForestClassifier loadedRf = PickleSerializer.fromPickle("random_forest_model.pkl");
            double loadedAccuracy = loadedRf.score(X, y);
            System.out.printf("   Loaded model accuracy: %.4f\n", loadedAccuracy);
            
            // 5. Export for Python
            System.out.println("\n5. Exporting for Python...");
            PythonBridge.exportToPython(rf, "random_forest_model");
            System.out.println("   Generated: random_forest_model.pkl");
            System.out.println("   Generated: random_forest_model_loader.py");
            System.out.println("   Generated: random_forest_model_metadata.json");
            
            // 6. Save as NumPy format
            System.out.println("\n6. Saving data as NumPy format...");
            NumPyBridge.saveAsNumpy(X, "data.npy");
            System.out.println("   Saved: data.npy");
            
            // 7. Load from NumPy
            System.out.println("\n7. Loading from NumPy format...");
            float[][] loadedX = NumPyBridge.loadFromNumpy("data.npy");
            System.out.printf("   Loaded data shape: %d x %d\n", loadedX.length, loadedX[0].length);
            
            // 8. Verify round-trip
            System.out.println("\n8. Verifying round-trip serialization...");
            boolean match = Arrays.deepEquals(X, loadedX);
            System.out.printf("   Data integrity: %s\n", match ? "PASSED" : "FAILED");
            
            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║                    ✓ SUCCESS!                                ║");
            System.out.println("║    Models can be saved and loaded across Java/Python        ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static float[][] generateSampleData(int nSamples) {
        float[][] X = new float[nSamples][4];
        Random rng = new Random(42);
        
        for (int i = 0; i < nSamples; i++) {
            int cluster = i % 3;
            double x1 = rng.nextGaussian() + (cluster == 0 ? -3 : cluster == 1 ? 0 : 3);
            double x2 = rng.nextGaussian() + (cluster == 0 ? -3 : cluster == 1 ? 3 : 0);
            
            X[i][0] = (float) (x1 + rng.nextGaussian() * 0.5);
            X[i][1] = (float) (x2 + rng.nextGaussian() * 0.5);
            X[i][2] = (float) rng.nextGaussian();
            X[i][3] = (float) rng.nextGaussian();
        }
        
        return X;
    }
    
    private static int[] generateSampleLabels(int nSamples) {
        int[] y = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            y[i] = i % 3;
        }
        return y;
    }
}


================================================
FILE: examples/jbang/common/PickleRunnerExample.java
================================================
package tech.kayys.tafkir.ml.example;

import tech.kayys.tafkir.ml.pickle.*;
import tech.kayys.tafkir.ml.base.*;
import java.util.*;

/**
 * Complete example demonstrating loading and running pickle models in Java.
 */
public class PickleRunnerExample {
    
    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║           Pickle Model Runner - Complete Example            ║");
        System.out.println("║      Load scikit-learn/PyTorch models in Java               ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝\n");
        
        try {
            // ============================================================
            // 1. Load Pickle Model
            // ============================================================
            System.out.println("1. Loading pickle model...");
            
            // Create pickle from Python first:
            // python -c "
            // import pickle
            // from sklearn.ensemble import RandomForestClassifier
            // import numpy as np
            // X = np.random.randn(100, 4)
            // y = np.random.randint(0, 3, 100)
            // model = RandomForestClassifier(n_estimators=10)
            // model.fit(X, y)
            // with open('model.pkl', 'wb') as f:
            //     pickle.dump(model, f)
            // "
            
            PickleModelRunner runner = new PickleModelRunner("model.pkl");
            runner.load();
            
            // Display metadata
            System.out.println("\n   Model Metadata:");
            for (var entry : runner.getMetadata().entrySet()) {
                System.out.printf("     %s: %s\n", entry.getKey(), entry.getValue());
            }
            
            // ============================================================
            // 2. Run Inference
            // ============================================================
            System.out.println("\n2. Running inference...");
            
            // Generate test samples
            Random rng = new Random(42);
            float[][] testSamples = new float[10][4];
            for (int i = 0; i < testSamples.length; i++) {
                for (int j = 0; j < 4; j++) {
                    testSamples[i][j] = (float) rng.nextGaussian();
                }
            }
            
            // Predict
            long startTime = System.currentTimeMillis();
            int[] predictions = runner.predictBatch(testSamples);
            long inferenceTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("   Inference completed in %d ms\n", inferenceTime);
            System.out.println("   Predictions: " + Arrays.toString(predictions));
            
            // ============================================================
            // 3. Async Inference
            // ============================================================
            System.out.println("\n3. Async inference demo...");
            
            var future = runner.predictAsync(testSamples);
            future.thenAccept(results -> {
                System.out.println("   Async results: " + Arrays.toString(results));
            }).get(); // Wait for completion
            
            // ============================================================
            // 4. Probability Predictions
            // ============================================================
            System.out.println("\n4. Getting probabilities...");
            
            double[][] probs = runner.predictProba(testSamples);
            System.out.printf("   Probabilities shape: %d x %d\n", probs.length, probs[0].length);
            for (int i = 0; i < Math.min(3, probs.length); i++) {
                System.out.printf("   Sample %d: ", i);
                for (double p : probs[i]) {
                    System.out.printf("%.3f ", p);
                }
                System.out.println();
            }
            
            // ============================================================
            // 5. Performance Statistics
            // ============================================================
            System.out.println("\n5. Performance Statistics:");
            var stats = runner.getStats();
            System.out.printf("   %s\n", stats);
            System.out.printf("   Throughput: %.1f inferences/second\n", stats.throughputPerSecond());
            
            // ============================================================
            // 6. Save as Tafkir Native Format
            // ============================================================
            System.out.println("\n6. Saving as Tafkir native format...");
            runner.saveAsTafkir("converted_model.tafkir");
            System.out.println("   Saved: converted_model.tafkir");
            
            // ============================================================
            // 7. Batch Processing with Different Sizes
            // ============================================================
            System.out.println("\n7. Batch size performance test:");
            
            int[] batchSizes = {1, 8, 32, 64, 128};
            for (int batchSize : batchSizes) {
                float[][] batch = new float[batchSize][4];
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < 4; j++) {
                        batch[i][j] = (float) rng.nextGaussian();
                    }
                }
                
                long batchStart = System.nanoTime();
                int[] results = runner.predictBatch(batch);
                long batchTime = System.nanoTime() - batchStart;
                
                double avgLatency = batchTime / (double) batchSize / 1_000_000.0;
                System.out.printf("   Batch size %3d: total=%.2fms, avg=%.3fms per sample\n",
                    batchSize, batchTime / 1_000_000.0, avgLatency);
            }
            
            // ============================================================
            // 8. Cleanup
            // ============================================================
            runner.close();
            
            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║                    ✓ SUCCESS!                                ║");
            System.out.println("║    Pickle model loaded and running in pure Java              ║");
            System.out.println("║    No Python required for inference!                        ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            
            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║  Note: Create a model.pkl file first using the Python script ║");
            System.out.println("║  shown in the comments above.                                ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
        }
    }
}


================================================
FILE: examples/jbang/common/train_cli.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import tech.kayys.tafkir.ml.nn.optim.*;

public class train_cli {
    public static void main(String[] args) {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 10;
        float lr = args.length > 1 ? Float.parseFloat(args[1]) : 0.001f;
        
        System.out.println("Training for " + epochs + " epochs");
        System.out.println("Learning rate: " + lr);
        
        Sequential model = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );
        
        Adam optimizer = new Adam(lr);
        
        for (int e = 1; e <= epochs; e++) {
            System.out.println("Epoch " + e + "...");
        }
        
        System.out.println("Done!");
    }
}



================================================
FILE: examples/jbang/common/train_model.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import tech.kayys.tafkir.ml.nn.loss.*;
import tech.kayys.tafkir.ml.nn.optim.*;
import tech.kayys.tafkir.ml.autograd.GradTensor;

public class train_model {
    public static void main(String[] args) {
        // Model
        Sequential model = new Sequential(
            new Linear(28 * 28, 128),
            new ReLU(),
            new Linear(128, 10)
        );
        
        // Loss and optimizer
        CrossEntropyLoss loss = new CrossEntropyLoss();
        Adam optimizer = new Adam(model.parameters(), 0.001f);
        
        // Training (pseudo-code)
        for (int epoch = 1; epoch <= 10; epoch++) {
            System.out.println("Epoch " + epoch);
        }
        
        System.out.println("Training complete!");
    }
}



================================================
FILE: examples/jbang/common/xor_training.java
================================================
//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import tech.kayys.tafkir.ml.nn.loss.*;
import tech.kayys.tafkir.ml.nn.optim.*;
import tech.kayys.tafkir.ml.autograd.GradTensor;

/**
 * A complete, runnable example of training a Neural Network to solve the XOR problem.
 * XOR is a classic non-linear problem that requires at least one hidden layer.
 */
public class xor_training {
    public static void main(String[] args) {
        System.out.println("=== Training Tafkir NN on XOR Problem ===");

        // 1. Prepare Data (XOR Input: [0,0], [0,1], [1,0], [1,1])
        float[] xData = {0, 0, 0, 1, 1, 0, 1, 1};
        float[] yData = {0, 1, 1, 0};
        
        GradTensor x = GradTensor.of(xData, 4, 2);
        GradTensor y = GradTensor.of(yData, 4, 1);

        // 2. Define Model (2 inputs -> 8 hidden -> 1 output)
        Sequential model = new Sequential(
            new Linear(2, 8),
            new ReLU(),
            new Linear(8, 1)
            // Note: Sigmoid is handled internally by BCEWithLogitsLoss
        );

        // 3. Loss and Optimizer
        BCEWithLogitsLoss criterion = new BCEWithLogitsLoss(); 
        SGD optimizer = new SGD(model.parameters(), 0.1f);

        // 4. Training Loop
        System.out.println("Starting training...");
        for (int epoch = 1; epoch <= 500; epoch++) {
            // Forward pass
            GradTensor output = model.forward(x);
            GradTensor loss = criterion.compute(output, y);

            // Backward pass
            model.zeroGrad();
            loss.backward();

            // Weight update
            optimizer.step();

            if (epoch % 50 == 0) {
                System.out.printf("Epoch [%d/500], Loss: %.4f%n", epoch, loss.item());
            }
        }

        // 5. Evaluation
        model.eval();
        GradTensor finalOutput = model.forward(x);
        System.out.println("\nFinal Predictions:");
        for (int i = 0; i < 4; i++) {
            float pred = finalOutput.data()[i];
            System.out.printf("Input: [%.0f, %.0f] -> Target: %.0f -> Predicted: %.4f%n", 
                xData[i*2], xData[i*2+1], yData[i], pred);
        }
    }
}



================================================
FILE: examples/jbang/edge/README.md
================================================
# Tafkir LiteRT Edge Examples

Enhanced examples demonstrating how to use LiteRT for edge device inference with JBang. These examples cover vision models (`.litert`) and language models (`.litertlm`) with comprehensive error handling, validation, and performance benchmarking.

## 🚀 Quick Start

### Prerequisites

- **Java 21 or higher** - [Download Adoptium](https://adoptium.net/)
- **JBang** - Install with: `curl -Ls https://sh.jbang.dev | bash -s -`
- **Maven** - For building the SDK modules
- **LiteRT Models** - `.litertlm` for language models, `.litert` for vision models

### Building the SDK

Before running the examples, build and install the LiteRT SDK modules:

```bash
# Navigate to the tafkir directory
cd tafkir

# Build and install the LiteRT SDK module
mvn clean install -pl sdk/lib/tafkir-sdk-litert -am -DskipTests

# Verify installation
ls ~/.m2/repository/tech/kayys/tafkir/tafkir-sdk-litert/
```

### Running Examples

**Using the enhanced runner script (recommended):**

```bash
# List available examples
./run.sh

# Run an example
./run.sh edge
./run.sh image --model mobilenet.litert --image cat.jpg
./run.sh text --model gemma.litertlm --prompt "Hello world"
./run.sh benchmark --model model.litert --iterations 100
```

**Using JBang directly:**

```bash
# Basic edge inference (demo mode)
jbang LiteRTEdgeExample.java

# With a real model
jbang -Dmodel=/path/to/model.litert LiteRTEdgeExample.java

# Image classification
jbang ImageClassificationExample.java \
  --model mobilenet_v2.litert \
  --image cat.jpg \
  --topk 5

# Object detection
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg \
  --threshold 0.5

# Text generation
jbang TextGenerationExample.java \
  --model gemma.litertlm \
  --prompt "Once upon a time" \
  --maxtokens 100 \
  --temperature 0.7

# Performance benchmark
jbang BenchmarkExample.java \
  --model model.litert \
  --iterations 100 \
  --batch 8 \
  --output results.csv
```

## 📚 Available Examples

### 1. LiteRTEdgeExample.java - Basic Edge Inference

**Purpose**: Demonstrates core LiteRT SDK capabilities including model loading, synchronous/asynchronous inference, batch processing, and performance metrics.

**Features**:
- ✅ Model loading and management
- ✅ Synchronous and asynchronous inference
- ✅ Batch processing
- ✅ Performance metrics collection
- ✅ Hardware delegate selection
- ✅ Comprehensive error handling

**Usage**:

```bash
# Demo mode (no model required)
jbang LiteRTEdgeExample.java

# With a language model
jbang -Dmodel=gemma-4-E4B-it-litert-lm.litertlm LiteRTEdgeExample.java

# With a vision model
jbang -Dmodel=mobilenet_v2.litert LiteRTEdgeExample.java

# Use GPU delegate
jbang -Dmodel=model.litert -Ddelegate=GPU LiteRTEdgeExample.java

# Custom configuration
jbang -Dmodel=model.litert -Dthreads=8 -Dxnnpack=true LiteRTEdgeExample.java
```

**Example Output**:
```
Configuration:
  Delegate:    AUTO
  Threads:     4
  XNNPACK:     true
  Memory Pool: true

✓ LiteRT SDK initialized
  Version: 0.1.0-SNAPSHOT
  Available: true

Loading model: mobilenet_v2.litert
✓ Model loaded

Model Information:
  ID:    edge-model
  Size:  13.45 MB
  Inputs: 1
  Outputs: 1

Running synchronous inference...
✓ Inference completed in 45ms
  Output size: 4000 bytes

Performance Metrics:
  Total inferences:    5
  Failed inferences:   0
  Avg latency:         42.35 ms
  P50 latency:         41.20 ms
  P95 latency:         48.50 ms
  P99 latency:         49.10 ms
  Peak memory:         45.67 MB
  Current memory:      32.12 MB
  Active delegate:     CPU
```

---

### 2. ImageClassificationExample.java - Image Classification

**Purpose**: Image classification using models like MobileNet, EfficientNet, ResNet, etc. with ImageNet 1000-class labels.

**Features**:
- ✅ Image preprocessing and resizing
- ✅ Single image classification
- ✅ Top-K predictions with ImageNet labels
- ✅ Batch classification with multiple crops
- ✅ Comprehensive error handling and validation

**Usage**:

```bash
# Basic classification
jbang ImageClassificationExample.java \
  --model mobilenet_v2.litert \
  --image cat.jpg

# Top-10 predictions
jbang ImageClassificationExample.java \
  --model mobilenet_v2.litert \
  --image cat.jpg \
  --topk 10

# Use GPU acceleration
jbang ImageClassificationExample.java \
  --model mobilenet_v2.litert \
  --image cat.jpg \
  --delegate GPU
```

**Example Output**:
```
Top 5 Predictions:
  1. Egyptian cat                  87.45%
  2. tabby                         8.32%
  3. tiger cat                     2.15%
  4. Siamese cat                   1.08%
  5. lynx                          0.52%
```

**Supported Models**:
- MobileNet V2/V3
- EfficientNet B0-B7
- ResNet-50/101
- Inception V3
- Any ImageNet-trained `.litert` classification model

---

### 3. ObjectDetectionExample.java - Object Detection

**Purpose**: Object detection using SSD MobileNet or YOLO models with bounding box visualization.

**Features**:
- ✅ Object detection with bounding boxes
- ✅ Confidence threshold filtering
- ✅ Visual output with drawn detections
- ✅ Batch detection with multiple thresholds
- ✅ COCO dataset labels (80 classes)
- ✅ Automatic output image generation

**Usage**:

```bash
# Basic detection
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg

# Custom confidence threshold
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg \
  --threshold 0.6

# Save output to specific file
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg \
  --output detected.jpg

# Use CoreML on iOS/macOS
jbang ObjectDetectionExample.java \
  --model ssd_mobilenet.litert \
  --image street.jpg \
  --delegate COREML
```

**Example Output**:
```
Detections:
  1. Class 3 (car) (92.34%) at [120, 80, 200, 150]
  2. Class 0 (person) (87.12%) at [300, 100, 100, 250]
  3. Class 16 (dog) (76.45%) at [50, 200, 80, 100]

Drawing detections on image...
  Saved to: street_detected.jpg
```

**Supported Models**:
- SSD MobileNet V1/V2
- YOLO V5/V8 (converted to LiteRT)
- Any COCO-trained `.litert` detection model

---

### 4. TextGenerationExample.java - Text Generation ⭐ NEW

**Purpose**: Text generation using language models like Gemma, Llama, Phi, etc.

**Features**:
- ✅ Text tokenization and preprocessing
- ✅ Single-turn text generation
- ✅ Interactive chat mode
- ✅ Configurable generation parameters (temperature, top-k, max tokens)
- ✅ Streaming generation simulation

**Usage**:

```bash
# Basic text generation
jbang TextGenerationExample.java \
  --model gemma-4-E4B-it-litert-lm.litertlm \
  --prompt "Hello, how are you?"

# With custom parameters
jbang TextGenerationExample.java \
  --model model.litertlm \
  --prompt "Write a story about" \
  --maxtokens 100 \
  --temperature 0.8 \
  --topk 50

# Interactive chat mode
jbang TextGenerationExample.java \
  --model model.litertlm \
  --interactive
```

**Example Output**:
```
Configuration:
  Model:       gemma-4-E4B-it-litert-lm.litertlm
  Prompt:      Hello, how are you?
  Max Tokens:  50
  Temperature: 0.7
  Top-K:       40
  Delegate:    AUTO

Loading model: gemma-4-E4B-it-litert-lm.litertlm
✓ Model loaded

Generating text...
Prompt: "Hello, how are you?"

Generated text:
I'm doing well, thank you for asking! How can I help you today?

✓ Generation completed in 234ms
  Tokens/sec: 213.68
```

**Supported Models**:
- Gemma 2B/7B
- Llama 2/3
- Microsoft Phi-2/3
- Any conversational `.litertlm` language model

---

### 5. BenchmarkExample.java - Performance Benchmarking ⭐ NEW

**Purpose**: Comprehensive performance benchmarking tool for measuring LiteRT inference performance.

**Features**:
- ✅ Single inference latency benchmark
- ✅ Batch inference throughput benchmark
- ✅ Async inference concurrency benchmark
- ✅ Sustained load testing
- ✅ Statistical analysis (mean, median, percentiles)
- ✅ CSV report generation
- ✅ Memory usage tracking

**Usage**:

```bash
# Basic benchmark
jbang BenchmarkExample.java \
  --model model.litert

# Custom benchmark parameters
jbang BenchmarkExample.java \
  --model model.litert \
  --iterations 100 \
  --batch 8 \
  --warmup 10

# With GPU delegate
jbang BenchmarkExample.java \
  --model model.litert \
  --delegate GPU \
  --threads 8

# Generate CSV report
jbang BenchmarkExample.java \
  --model model.litert \
  --output results.csv
```

**Example Output**:
```
Benchmark Configuration:
  Model:         model.litert
  Iterations:    50
  Batch Size:    1
  Warmup Runs:   10
  Delegate:      AUTO
  Threads:       4
  XNNPACK:       true

══════════════════════════════════════════════════════════
Benchmark 1: Single Inference Latency
══════════════════════════════════════════════════════════
  Test:            Single Inference
  Mean Latency:    42.35 ms
  Median Latency:  41.20 ms
  P50 Latency:     41.20 ms
  P95 Latency:     48.50 ms
  P99 Latency:     49.10 ms
  Min Latency:     38 ms
  Max Latency:     52 ms
  Std Deviation:   3.45 ms
  Throughput:      23.61 inf/sec
  Peak Memory:     45.67 MB

══════════════════════════════════════════════════════════
Benchmark Summary
══════════════════════════════════════════════════════════
  Single Inference                        42.35 ms (mean)     23.61 inf/sec
  Batch Inference (batch=8)               156.78 ms (mean)    51.03 inf/sec
  Async Inference (concurrency=8)         89.45 ms (mean)     11.18 inf/sec
  Sustained Load (60s)                    43.12 ms (mean)     23.19 inf/sec
```

**CSV Report Format**:
```csv
Test,Mean_ms,Median_ms,P50_ms,P95_ms,P99_ms,Min_ms,Max_ms,StdDev_ms,Throughput_inf_sec,Peak_Memory_B
Sing