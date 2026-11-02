package tech.kayys.tafkir.integration.aljabr.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.safetensor.quantization.QuantConfig;
import tech.kayys.aljabr.safetensor.quantization.QuantizationEngine.QuantStrategy;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** Prevents unvalidated adapters from being selected or called directly. */
class QuantizerAvailabilityTest {
    @Test
    void incompleteAdaptersRemainUnavailable() {
        assertFalse(new GPTQQuantizerAdapter().supports(QuantConfig.int4Gptq()));
        assertFalse(new AWQQuantizerAdapter().supports(QuantConfig.awq(4, 128)));
        assertFalse(new AutoRoundQuantizerAdapter().supports(QuantConfig.autoRound()));
        assertFalse(new TurboQuantAdapter().supports(QuantConfig.turbo()));
        assertThrows(UnsupportedOperationException.class,
                () -> new GPTQQuantizerAdapter().quantizeTensor(null, QuantConfig.int4Gptq()));
        assertThrows(UnsupportedOperationException.class,
                () -> new AWQQuantizerAdapter().quantizeTensor(null, QuantConfig.awq(4, 128)));
        assertThrows(UnsupportedOperationException.class,
                () -> new AutoRoundQuantizerAdapter().quantizeTensor(null, QuantConfig.autoRound()));
        assertThrows(UnsupportedOperationException.class,
                () -> new TurboQuantAdapter().quantizeTensor(null, QuantConfig.turbo()));
        assertThrows(UnsupportedOperationException.class,
                () -> new GPTQQuantizerAdapter().dequantizeTensor(null, QuantConfig.int4Gptq()));
        assertThrows(UnsupportedOperationException.class,
                () -> new AWQQuantizerAdapter().dequantizeTensor(null, QuantConfig.awq(4, 128)));
        assertThrows(UnsupportedOperationException.class,
                () -> new AutoRoundQuantizerAdapter().dequantizeTensor(null, QuantConfig.autoRound()));
        assertThrows(UnsupportedOperationException.class,
                () -> new TurboQuantAdapter().dequantizeTensor(null, QuantConfig.turbo()));
    }

    @Test
    void bitsAndBytesAdvertisesOnlyItsImplementedNf4Path() {
        BnBQuantizerAdapter adapter = new BnBQuantizerAdapter();
        assertTrue(adapter.supports(QuantConfig.bnb()));
        assertFalse(adapter.supports(QuantConfig.int8()));
        assertFalse(adapter.supports(QuantConfig.builder().strategy(QuantStrategy.BNB).bits(8).build()));
    }
}
