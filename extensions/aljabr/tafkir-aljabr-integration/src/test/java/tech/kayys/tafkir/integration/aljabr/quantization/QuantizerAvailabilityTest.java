package tech.kayys.tafkir.integration.aljabr.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.safetensor.quantization.QuantConfig;
import tech.kayys.aljabr.safetensor.quantization.QuantizationEngine.QuantStrategy;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Prevents incomplete adapters from being selected by Aljabr. */
class QuantizerAvailabilityTest {
    @Test
    void incompleteAdaptersRemainUnavailable() {
        assertFalse(new GPTQQuantizerAdapter().supports(QuantConfig.int4Gptq()));
        assertFalse(new AWQQuantizerAdapter().supports(QuantConfig.awq(4, 128)));
        assertFalse(new AutoRoundQuantizerAdapter().supports(QuantConfig.autoRound()));
        assertFalse(new TurboQuantAdapter().supports(QuantConfig.turbo()));
    }

    @Test
    void bitsAndBytesAdvertisesOnlyItsImplementedNf4Path() {
        BnBQuantizerAdapter adapter = new BnBQuantizerAdapter();
        assertTrue(adapter.supports(QuantConfig.bnb()));
        assertFalse(adapter.supports(QuantConfig.int8()));
        assertFalse(adapter.supports(QuantConfig.builder().strategy(QuantStrategy.BNB).bits(8).build());
    }
}
