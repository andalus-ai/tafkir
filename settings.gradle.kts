rootProject.name = "tafkir-engine"

// Composite build resolution for Aljabr foundation
val aljabrDir = listOf(
    file("../../Aljabr/aljabr"),
    file("../../aljabr"),
    file("../aljabr")
).firstOrNull { it.isDirectory && (it.resolve("build.gradle.kts").isFile || it.resolve("settings.gradle.kts").isFile) }

if (aljabrDir != null) {
    includeBuild(aljabrDir.absolutePath)
}

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

fun includeLegacyAlias(projectPath: String, projectDir: String) {
    if (!file(projectDir).isDirectory) return
    include(projectPath)
    project(":$projectPath").projectDir = file(projectDir)
}

val staticallyIncludedModelProjects = setOf<String>()

includeOptionalProject("suling", "../extensions/audio/suling")

// Production modules
includeLegacyAlias("compiler:tafkir-compiler", "compiler/tafkir-compiler")
includeLegacyAlias("compiler:tafkir-neural-compiler", "compiler/tafkir-neural-compiler")

includeLegacyAlias("data:core:tafkir-data", "data/core/tafkir-data")
includeLegacyAlias("data:core:tafkir-learn-data", "data/core/tafkir-learn-data")
includeLegacyAlias("data:core:tafkir-train-data", "data/core/tafkir-train-data")

includeLegacyAlias("runtime:core:tafkir-distributed", "runtime/core/tafkir-distributed")
includeLegacyAlias("runtime:core:tafkir-error-code", "runtime/core/tafkir-error-code")
includeLegacyAlias("runtime:autograd:tafkir-ml-autograd", "runtime/autograd/tafkir-ml-autograd")
includeLegacyAlias("runtime:serialization:tafkir-serializer", "runtime/serialization/tafkir-serializer")

includeLegacyAlias("training:api:tafkir-learn-api", "training/api/tafkir-learn-api")
includeLegacyAlias("training:api:tafkir-train-diffusion-api", "training/api/tafkir-train-diffusion-api")
includeLegacyAlias("training:api:tafkir-trainer-api", "training/api/tafkir-trainer-api")
includeLegacyAlias("training:loop:tafkir-learn-trainer", "training/loop/tafkir-learn-trainer")
includeLegacyAlias("training:loop:tafkir-train-diffusion-opd", "training/loop/tafkir-train-diffusion-opd")
includeLegacyAlias("training:loop:tafkir-trainer", "training/loop/tafkir-trainer")
includeLegacyAlias("training:optim:tafkir-learn-optim", "training/optim/tafkir-learn-optim")
includeLegacyAlias("training:strategy:tafkir-train-strategy", "training/strategy/tafkir-train-strategy")

includeLegacyAlias("ml:core:tafkir-ml-byte-latent", "ml/core/tafkir-ml-byte-latent")
includeLegacyAlias("ml:core:tafkir-ml-core", "ml/core/tafkir-ml-core")
includeLegacyAlias("ml:core:tafkir-ml-hub", "ml/core/tafkir-ml-hub")
includeLegacyAlias("ml:core:tafkir-ml-persistence", "ml/core/tafkir-ml-persistence")
includeLegacyAlias("ml:core:tafkir-ml-timeseries", "ml/core/tafkir-ml-timeseries")
includeLegacyAlias("ml:core:tafkir-train-estimator", "ml/core/tafkir-train-estimator")
includeLegacyAlias("ml:core:tafkir-train-recursive-reasoning", "ml/core/tafkir-train-recursive-reasoning")
includeLegacyAlias("ml:selection:tafkir-ml-selection", "ml/selection/tafkir-ml-selection")

includeLegacyAlias("models:nn-audio:tafkir-nn-audio", "models/nn-audio/tafkir-nn-audio")
includeLegacyAlias("models:nn-core:tafkir-learn-nn", "models/nn-core/tafkir-learn-nn")
includeLegacyAlias("models:nn-core:tafkir-nn", "models/nn-core/tafkir-nn")
includeLegacyAlias("models:nn-core:tafkir-nn-optimize", "models/nn-core/tafkir-nn-optimize")
includeLegacyAlias("models:nn-multimodal:tafkir-nn-multimodal", "models/nn-multimodal/tafkir-nn-multimodal")
includeLegacyAlias("models:nn-rnn:tafkir-nn-rnn", "models/nn-rnn/tafkir-nn-rnn")
includeLegacyAlias("models:nn-vision:tafkir-nn-cnn", "models/nn-vision/tafkir-nn-cnn")
includeLegacyAlias("models:nn-vision:tafkir-nn-vision", "models/nn-vision/tafkir-nn-vision")

includeLegacyAlias("nlp:core:tafkir-nlp", "nlp/core/tafkir-nlp")
includeLegacyAlias("nlp:transformers:tafkir-train-transformer", "nlp/transformers/tafkir-train-transformer")

includeLegacyAlias("inference:core:tafkir-ml-runner-api", "inference/core/tafkir-ml-runner-api")
includeLegacyAlias("inference:export:tafkir-ml-export", "inference/export/tafkir-ml-export")
includeLegacyAlias("inference:quantization:tafkir-quantizer-autoround", "inference/quantization/tafkir-quantizer-autoround")
includeLegacyAlias("inference:quantization:tafkir-quantizer-awq", "inference/quantization/tafkir-quantizer-awq")
includeLegacyAlias("inference:quantization:tafkir-quantizer-gptq", "inference/quantization/tafkir-quantizer-gptq")
includeLegacyAlias("inference:quantization:tafkir-quantizer-quip", "inference/quantization/tafkir-quantizer-quip")
includeLegacyAlias("inference:quantization:tafkir-quantizer-turboquant", "inference/quantization/tafkir-quantizer-turboquant")

includeLegacyAlias("ecosystem:cli:tafkir-cli", "ecosystem/cli/tafkir-cli")
includeLegacyAlias("ecosystem:examples:tafkir-train-examples", "ecosystem/examples/tafkir-train-examples")
includeLegacyAlias("ecosystem:jupyter:tafkir-jupyter-kernel", "ecosystem/jupyter/tafkir-jupyter-kernel")
includeLegacyAlias("ecosystem:integrations:integration:tafkir-langchain4j", "ecosystem/integrations/integration/langchain4j/tafkir-langchain4j")

includeLegacyAlias("extensions:aljabr:tafkir-ml-aljabr", "extensions/aljabr/tafkir-ml-aljabr")
includeLegacyAlias("extensions:aljabr:tafkir-aljabr-integration", "extensions/aljabr/tafkir-aljabr-integration")
includeLegacyAlias("extensions:aljabr:tafkir-trainer-aljabr", "extensions/aljabr/tafkir-trainer-aljabr")

// Compatibility aliases kept for legacy names that still show up in tooling or scripts.
includeLegacyAlias("quantizer:tafkir-quantizer-autoround", "inference/quantization/tafkir-quantizer-autoround")
includeLegacyAlias("quantizer:tafkir-quantizer-awq", "inference/quantization/tafkir-quantizer-awq")
includeLegacyAlias("quantizer:tafkir-quantizer-gptq", "inference/quantization/tafkir-quantizer-gptq")
includeLegacyAlias("quantizer:tafkir-quantizer-quip", "inference/quantization/tafkir-quantizer-quip")
includeLegacyAlias("quantizer:tafkir-quantizer-turboquant", "inference/quantization/tafkir-quantizer-turboquant")

includeLegacyAlias("ml:tafkir-ml-api", "training/api/tafkir-learn-api")
includeLegacyAlias("ml:tafkir-ml-aljabr", "extensions/aljabr/tafkir-aljabr-integration")
includeLegacyAlias("ml:tafkir-ml-autograd", "runtime/autograd/tafkir-ml-autograd")
includeLegacyAlias("ml:tafkir-ml-cnn", "models/nn-vision/tafkir-nn-cnn")
includeLegacyAlias("ml:tafkir-ml-core", "ml/core/tafkir-ml-core")
includeLegacyAlias("ml:tafkir-ml-data", "data/core/tafkir-learn-data")
includeLegacyAlias("ml:tafkir-ml-diffusion-api", "training/api/tafkir-train-diffusion-api")
includeLegacyAlias("ml:tafkir-ml-diffusion-opd", "training/loop/tafkir-train-diffusion-opd")
includeLegacyAlias("ml:tafkir-ml-estimator", "ml/core/tafkir-train-estimator")
includeLegacyAlias("ml:tafkir-ml-export", "inference/export/tafkir-ml-export")
includeLegacyAlias("ml:tafkir-ml-hub", "ml/core/tafkir-ml-hub")
includeLegacyAlias("ml:tafkir-ml-multimodal", "models/nn-multimodal/tafkir-nn-multimodal")
includeLegacyAlias("ml:tafkir-ml-nlp", "nlp/core/tafkir-nlp")
includeLegacyAlias("ml:tafkir-ml-nn", "models/nn-core/tafkir-learn-nn")
includeLegacyAlias("ml:tafkir-ml-optimize", "models/nn-core/tafkir-nn-optimize")
includeLegacyAlias("ml:tafkir-ml-persistence", "ml/core/tafkir-ml-persistence")
includeLegacyAlias("ml:tafkir-ml-preprocessing", "ml/preprocessing")
includeLegacyAlias("ml:tafkir-ml-runner-api", "inference/core/tafkir-ml-runner-api")
includeLegacyAlias("ml:tafkir-ml-selection", "ml/selection/tafkir-ml-selection")
includeLegacyAlias("ml:tafkir-ml-timeseries", "ml/core/tafkir-ml-timeseries")

val legacyModelFamilyCatalog = file("ecosystem/cli/tafkir-cli/build.gradle.kts")
    .takeIf { it.isFile }
    ?.readText()
    ?.let { text ->
        Regex("ModelFamilyModule\\(\\\"([^\\\"]+)\\\",\\s*\\\":models:tafkir-model-([^\\\"]+)\\\"")
            .findAll(text)
            .map { match -> match.groupValues[2] }
            .distinct()
            .toList()
    }
    ?: emptyList()

legacyModelFamilyCatalog.forEach { legacyModelName ->
    includeLegacyAlias("models:tafkir-model-$legacyModelName", "models/nn-core")
}

includeLegacyAlias("runtime:tafkir-runtime", "runtime/core/tafkir-distributed")
includeLegacyAlias("autograd", "runtime/autograd/tafkir-ml-autograd")

includeLegacyAlias("training:tafkir-learn-data", "data/core/tafkir-learn-data")
includeLegacyAlias("training:tafkir-learn-nn", "models/nn-core/tafkir-learn-nn")
includeLegacyAlias("training:tafkir-learn-optim", "training/optim/tafkir-learn-optim")
includeLegacyAlias("training:tafkir-learn-trainer", "training/loop/tafkir-learn-trainer")
includeLegacyAlias("trainer:tafkir-trainer-api", "training/api/tafkir-trainer-api")
