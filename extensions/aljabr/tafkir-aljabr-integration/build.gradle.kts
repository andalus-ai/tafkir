plugins {
    `java-library`
}

description = "Optional Tafkir quantizer providers for the Aljabr SafeTensors quantization SPI"

dependencies {
    // Aljabr owns the tensor-level SPI; this extension is the optional implementation bridge.
    api("tech.kayys.aljabr:aljabr-safetensor-quantization:0.1.0-SNAPSHOT")

    implementation(project(":inference:quantization:tafkir-quantizer-gptq"))
    implementation(project(":inference:quantization:tafkir-quantizer-awq"))
    implementation(project(":inference:quantization:tafkir-quantizer-autoround"))
    implementation(project(":inference:quantization:tafkir-quantizer-turboquant"))
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
}

java {
    withSourcesJar()
}
