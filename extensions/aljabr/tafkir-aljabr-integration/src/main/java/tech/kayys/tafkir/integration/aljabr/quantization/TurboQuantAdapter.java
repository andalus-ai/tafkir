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
        log.infof("TurboQuant: dequantizing tensor %s", Arrays.toString(quantizedTensor.shape()));
        
        int dim = (int) quantizedTensor.numel();
        byte[] packed = quantizedTensor.dataSegment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        int[] indices = new int[dim];
        for (int i = 0; i < dim; i++) {
            indices[i] = packed[i] & 0xFF;
        }
        
        TurboQuantConfig tqConfig = new TurboQuantConfig(
            config.getBits(),
            dim,
            TurboQuantConfig.Variant.MSE,
            TurboQuantConfig.RotationStrategy.HADAMARD,
            false,
            0,
            42L
        );
        
        TurboQuantEngine engine = new TurboQuantEngine(tqConfig);
        float[] output = new float[dim];
        engine.dequantizeMse(indices, output);
        
        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
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
