plugins {
    java
    `maven-publish`
}

group = "tech.kayys.tafkir"
version = "0.3.0-SNAPSHOT"

dependencies {
    implementation(project(":ml:tafkir-ml-aljabr"))
    implementation(project(":trainer:tafkir-trainer-api"))
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-nn:0.1.0-SNAPSHOT")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

tasks.withType<JavaCompile> {
    options.compilerArgs.add("--enable-preview")
    options.compilerArgs.add("--add-modules=jdk.incubator.vector")
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            groupId = "tech.kayys.tafkir"
            artifactId = "tafkir-trainer-aljabr"
            version = "0.3.0-SNAPSHOT"
        }
    }
}
