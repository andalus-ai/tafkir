# Published Aljabr/Tafkir consumer smoke test

This fixture is intentionally a separate Gradle build. It does not include either
repository as a composite build and does not use project dependencies. It resolves
Aljabr and Tafkir exclusively as Maven artifacts, then verifies that Aljabr's
`QuantizerRegistry` discovers all five optional Tafkir quantizer providers.

## Prerequisites

- JDK 25
- A local Maven repository (`~/.m2/repository`)
- The Aljabr and Tafkir source checkouts

## Publish the required artifacts locally

From the Aljabr checkout, publish the SafeTensors modules and their project
dependencies. The exact task set may need to be expanded if the POM gains new
project dependencies:

```bash
./gradlew \
  :core:aljabr-safetensor-api:publishToMavenLocal \
  :core:aljabr-safetensor-spi:publishToMavenLocal \
  :core:aljabr-spi-tokenizer:publishToMavenLocal \
  :core:aljabr-safetensor-core:publishToMavenLocal \
  :core:aljabr-safetensor-bridge:publishToMavenLocal \
  :core:aljabr-safetensor-loader:publishToMavenLocal \
  :core:aljabr-safetensor-quantization:publishToMavenLocal
```

From the Tafkir checkout, publish the quantizer modules and optional bridge:

```bash
./gradlew \
  :inference:quantization:tafkir-quantizer-gptq:publishToMavenLocal \
  :inference:quantization:tafkir-quantizer-awq:publishToMavenLocal \
  :inference:quantization:tafkir-quantizer-autoround:publishToMavenLocal \
  :inference:quantization:tafkir-quantizer-turboquant:publishToMavenLocal \
  :extensions:aljabr:tafkir-aljabr-integration:publishToMavenLocal
```

If a module is not publishable or a POM dependency is missing, fix publication
metadata rather than adding a composite-build substitution to this fixture.
This test is intended to expose packaging problems.

## Run the external consumer

From the Tafkir repository root:

```bash
./gradlew --no-daemon -p integration-tests/aljabr-published-consumer clean verifyPublishedIntegration
```

The smoke test passes only when the published Aljabr registry discovers all five
Tafkir providers from the published runtime artifact. It does not assert that
the quantization algorithms are numerically correct; algorithm validation belongs
in their respective module tests.
