package tech.kayys.tafkir.integration.aljabr.quantization;

import tech.kayys.aljabr.safetensor.quantization.quantizer.Quantizer;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.quantization.QuantConfig;
import tech.kayys.tafkir.quantizer.autoround.AutoRoundDequantizer;
import tech.kayys.tafkir.quantizer.autoround.AutoRoundConfig;
import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Adapter for AutoRound quantization.
 * @author bhangun
 */
public class AutoRoundQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(AutoRoundQuantizerAdapter.class);

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        throw new UnsupportedOperationException("AutoRound adapter is disabled until its algorithm has reference-backed numerical validation.");
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        throw new UnsupportedOperationException(
                "AutoRound adapter is disabled until its serialized format and numerical behavior are reference-validated.");
    }

    @Override
    public String getName() {
        return "AutoRound";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return false; // Deliberately unavailable until validated against a reference implementation.
    }
}
