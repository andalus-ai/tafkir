import org.gradle.api.GradleException

/**
 * Checks that Tafkir's optional Aljabr bridge remains a provider-only extension:
 * Aljabr contracts are compile APIs, Tafkir quantizers are implementations, and
 * the Java service descriptor names real provider classes.
 */
val verifyAljabrIntegrationBoundary = tasks.register("verifyAljabrIntegrationBoundary") {
    group = "verification"
    description = "Verifies Tafkir's Aljabr quantizer SPI dependencies and ServiceLoader providers."

    doLast {
        val integrationDir = rootProject.file("extensions/aljabr/tafkir-aljabr-integration")
        val buildFile = integrationDir.resolve("build.gradle.kts")
        val descriptor = integrationDir.resolve(
            "src/main/resources/META-INF/services/" +
                "tech.kayys.aljabr.safetensor.quantization.quantizer.Quantizer"
        )
        val errors = mutableListOf<String>()

        if (!buildFile.isFile) {
            errors += "Missing Aljabr integration build file: ${buildFile.relativeTo(rootProject.projectDir)}"
        }
        if (!descriptor.isFile) {
            errors += "Missing Quantizer ServiceLoader descriptor: ${descriptor.relativeTo(rootProject.projectDir)}"
        }

        if (buildFile.isFile) {
            val buildText = buildFile.readText()
            listOf(
                """api("tech.kayys.aljabr:aljabr-safetensor-quantization:""",
                """api("tech.kayys.aljabr:aljabr-safetensor-core:""",
                """implementation(project(":inference:quantization:tafkir-quantizer-gptq"))""",
                """implementation(project(":inference:quantization:tafkir-quantizer-awq"))""",
                """implementation(project(":inference:quantization:tafkir-quantizer-autoround"))""",
                """implementation(project(":inference:quantization:tafkir-quantizer-turboquant"))"""
            ).filterNot(buildText::contains).forEach {
                errors += "Integration module is missing required dependency declaration: $it"
            }
        }

        if (descriptor.isFile) {
            val providers = descriptor.readLines()
                .map { it.substringBefore("#").trim() }
                .filter { it.isNotEmpty() }
            if (providers.isEmpty()) errors += "ServiceLoader descriptor has no providers."

            providers.forEach { provider ->
                if (!provider.startsWith("tech.kayys.tafkir.integration.aljabr.quantization.")) {
                    errors += "Provider is outside Tafkir's integration package: $provider"
                }
                val sourcePath = provider.replace('.', '/') + ".java"
                val source = rootProject.file("extensions/aljabr/tafkir-aljabr-integration/src/main/java/$sourcePath")
                if (!source.isFile) {
                    errors += "ServiceLoader provider source does not exist: $sourcePath"
                } else if (!Regex("""implements\s+Quantizer""").containsMatchIn(source.readText())) {
                    errors += "Provider does not implement Aljabr's Quantizer SPI: $provider"
                }
            }

            val expectedProviders = setOf(
                "tech.kayys.tafkir.integration.aljabr.quantization.GPTQQuantizerAdapter",
                "tech.kayys.tafkir.integration.aljabr.quantization.AWQQuantizerAdapter",
                "tech.kayys.tafkir.integration.aljabr.quantization.AutoRoundQuantizerAdapter",
                "tech.kayys.tafkir.integration.aljabr.quantization.BnBQuantizerAdapter",
                "tech.kayys.tafkir.integration.aljabr.quantization.TurboQuantAdapter"
            )
            if (providers.toSet() != expectedProviders || providers.size != expectedProviders.size) {
                errors += "Provider descriptor differs from the expected five Tafkir quantizer adapters."
            }
        }

        if (errors.isNotEmpty()) {
            throw GradleException(
                "Tafkir/Aljabr integration boundary verification failed:\n" +
                    errors.joinToString("\n") { " - $it" }
            )
        }
        logger.lifecycle("Tafkir/Aljabr integration boundary verified: Aljabr SPI and five providers are registered.")
    }
}

allprojects {
    tasks.matching { it.name == "check" }.configureEach {
        dependsOn(verifyAljabrIntegrationBoundary)
    }
}
