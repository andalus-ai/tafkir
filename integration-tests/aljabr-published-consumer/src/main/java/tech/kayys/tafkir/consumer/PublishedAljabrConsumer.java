package tech.kayys.tafkir.consumer;

import tech.kayys.aljabr.safetensor.quantization.QuantizerRegistry;

import java.util.Set;

/**
 * External-consumer smoke test. This source compiles against Aljabr's published
 * API and discovers Tafkir providers only from the runtime Maven artifact.
 */
public final class PublishedAljabrConsumer {
    private static final Set<String> EXPECTED_PROVIDERS = Set.of(
            "gptqquantizeradapter",
            "awqquantizeradapter",
            "autoroundquantizeradapter",
            "bnbquantizeradapter",
            "turboquantadapter"
    );

    private PublishedAljabrConsumer() {
    }

    public static void main(String[] args) {
        Set<String> discovered = QuantizerRegistry.getNames();
        if (!discovered.containsAll(EXPECTED_PROVIDERS)) {
            Set<String> missing = new java.util.TreeSet<>(EXPECTED_PROVIDERS);
            missing.removeAll(discovered);
            throw new IllegalStateException(
                    "Published Tafkir integration did not register all providers. Missing: " + missing
                            + "; discovered: " + discovered
            );
        }

        System.out.println("Published Aljabr/Tafkir integration verified.");
        System.out.println("Discovered providers: " + EXPECTED_PROVIDERS);
    }
}
