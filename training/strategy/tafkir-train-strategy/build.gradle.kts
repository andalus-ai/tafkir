plugins {
    java
}

dependencies {
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")

    // The in-process off-heap provider is an optional Aljabr capability used
    // only by this integration test; it is not a Tafkir runtime dependency.
    testImplementation("tech.kayys.aljabr:aljabr-offheap:0.1.0-SNAPSHOT")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}
