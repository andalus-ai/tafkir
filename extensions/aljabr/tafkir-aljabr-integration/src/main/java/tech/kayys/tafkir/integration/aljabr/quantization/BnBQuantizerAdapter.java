package tech.kayys.tafkir.integration.aljabr.quantization;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.quantization.QuantConfig;
import tech.kayys.aljabr.safetensor.quantization.quantizer.Quantizer;

/**
 * BitsAndBytes NF4 integration placeholder.
 *
 * <p>This provider remains unavailable until its packed representation, metadata
 * lifecycle, and numerical behavior are validated against bitsandbytes reference
 * fixtures. In particular, the current in-repository encoder/decoder uses a
 * high-nibble-first convention, which is not the documented bitsandbytes
 * low-nibble-first byte layout. Failing closed prevents incompatible weights
 * from being emitted or consumed under the BitsAndBytes name.
 */
public final class BnBQuantizerAdapter implements Quantizer {

    private static UnsupportedOperationException unavailable() {
        return new UnsupportedOperationException(
                "BitsAndBytes NF4 adapter is disabled until packed-format and numerical parity are validated.");
    }

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        throw unavailable();
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        throw unavailable();
    }

    @Override
    public String getName() {
        return "BitsAndBytes";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return false;
    }
}
