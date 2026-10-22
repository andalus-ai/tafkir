plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.tafkir"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-tokenizer:0.1.0-SNAPSHOT")
    implementation(project(":ml:tafkir-ml-api"))
    implementation(project(":ml:tafkir-ml-autograd"))
    implementation(project(":ml:tafkir-ml-diffusion-opd"))
    implementation(project(":ml:tafkir-ml-nn"))
    implementation(project(":ml:tafkir-ml-cnn"))
    implementation(project(":ml:tafkir-ml-optimize"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-backend-metal:0.1.0-SNAPSHOT")
    implementation(project(":runner:alkhawarizm-diffusion"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-safetensor-loader:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-safetensor-quantization:0.1.0-SNAPSHOT")
    implementation(project(":runner:safetensor:alkhawarizm-runner-stable-diffusion"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-safetensor-core:0.1.0-SNAPSHOT")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.assertj", name = "assertj-core")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
