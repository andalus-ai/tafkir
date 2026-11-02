plugins {
    application
}

repositories {
    mavenLocal()
    mavenCentral()
}

dependencies {
    // Resolve only published Maven coordinates: no project dependencies or composite builds.
    implementation("tech.kayys.aljabr:aljabr-safetensor-quantization:0.1.0-SNAPSHOT")
    runtimeOnly("tech.kayys.tafkir:tafkir-aljabr-integration:0.1.1")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

application {
    mainClass.set("tech.kayys.tafkir.consumer.PublishedAljabrConsumer")
}

tasks.register<JavaExec>("verifyPublishedIntegration") {
    group = "verification"
    description = "Runs the consumer against independently published Aljabr and Tafkir artifacts."
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set(application.mainClass)
}
