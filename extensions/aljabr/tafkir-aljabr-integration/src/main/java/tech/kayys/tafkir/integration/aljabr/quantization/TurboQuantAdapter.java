package tech.kayys.tafkir.integration.aljabr.quantization;

import tech.kayys.aljabr.safetensor.quantization.quantizer.Quantizer;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.quantization.QuantConfig;
import tech.kayys.tafkir.quantizer.turboquant.TurboQuantEngine;
import tech.kayys.tafkir.quantizer.turboquant.TurboQuantConfig;
import org.jboss.logging.Logger;

import java.util.Arrays;

/**
 * Adapter that bridges the optimized TurboQuant core engine with the Safetensor runner.
 * @author bhangun
 */
public class TurboQuantAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(TurboQuantAdapter.class);

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        throw new UnsupportedOperationException("TurboQuant adapter is disabled until its algorithm has reference-backed numerical validation.");
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        throw new UnsupportedOperationException(
                "TurboQuant adapter is disabled until its serialized format and numerical behavior are reference-validated.");
    }

    @Override
    public String getName() {
        return "TurboQuant";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return false; // Deliberately unavailable until validated against a reference implementation.
    }
}
