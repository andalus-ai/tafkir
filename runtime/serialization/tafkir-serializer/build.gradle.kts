plugins {
    java
}

dependencies {

   // implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation(project(":ml:tafkir-ml-core"))
    implementation(project(":ml:tafkir-ml-persistence"))
    implementation(project(":ml:tafkir-ml-estimator"))
    implementation("com.google.code.gson:gson:2.10.1")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
