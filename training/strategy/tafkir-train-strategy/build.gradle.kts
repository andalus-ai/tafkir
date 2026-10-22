plugins {
    java
}

dependencies {
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
}

dependencies {
    testImplementation("tech.kayys.aljabr:aljabr-rocksdb:0.1.0-SNAPSHOT")
    testImplementation("tech.kayys.aljabr:aljabr-helixdb:0.1.0-SNAPSHOT")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}
