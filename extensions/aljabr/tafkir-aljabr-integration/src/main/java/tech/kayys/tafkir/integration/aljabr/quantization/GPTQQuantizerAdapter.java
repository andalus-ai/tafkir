package tech.kayys.tafkir.integration.aljabr.quantization;

import tech.kayys.aljabr.safetensor.quantization.quantizer.Quantizer;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.quantization.QuantConfig;
import tech.kayys.tafkir.quantizer.gptq.VectorDequantizer;
import tech.kayys.tafkir.quantizer.gptq.GPTQConfig;
import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Adapter for GPTQ quantization.
 * @author bhangun
 */
public class GPTQQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(GPTQQuantizerAdapter.class);

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        throw new UnsupportedOperationException("GPTQ adapter is disabled until its algorithm has reference-backed numerical validation.");
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        int inF = (int) quantizedTensor.size(0);
        int outF = (int) quantizedTensor.size(1);
        int bits = config.getBits();
        int groupSize = quantizedTensor.groupSize();
        
        GPTQConfig gptqConfig = new GPTQConfig(bits, groupSize, false, false, false, "float32", true, 0.01, 128, 2048, false, null);
        VectorDequantizer dequantizer = new VectorDequantizer(gptqConfig);
        
        MemorySegment dataSeg = quantizedTensor.dataSegment();
        int[] qweightInts = new int[(int) (dataSeg.byteSize() / 4)];
        for (int i = 0; i < qweightInts.length; i++) {
            qweightInts[i] = dataSeg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), (long) i * 4);
        }
        
        float[] scales = quantizedTensor.scales().toArray(ValueLayout.JAVA_FLOAT);
        float[] zeros = quantizedTensor.zeros().toArray(ValueLayout.JAVA_FLOAT);
        
        // Convert scales to short[] (FP16) as expected by VectorDequantizer
        short[] scalesShorts = new short[scales.length];
        for (int i = 0; i < scales.length; i++) scalesShorts[i] = (short) Float.floatToFloat16(scales[i]);
        
        // Unpack zeros to packed int[] if needed, or core might have a helper
        // VectorDequantizer.dequantize expects packed qzerosInts
        int[] qzerosInts = packZeros(zeros, (inF + groupSize - 1) / groupSize, outF, bits);
        
        float[] output = new float[inF * outF];
        dequantizer.dequantize(qweightInts, qzerosInts, scalesShorts, null, inF, outF, output);
        
        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
    }

    private int[] packZeros(float[] zeros, int numGroups, int outF, int bits) {
        int packFactor = 32 / bits;
        int[] packed = new int[numGroups * (outF / packFactor)];
        for (int g = 0; g < numGroups; g++) {
            for (int pj = 0; pj < outF / packFactor; pj++) {
                int word = 0;
                for (int b = 0; b < packFactor; b++) {
                    int j = pj * packFactor + b;
                    int z = (int) zeros[g * outF + j] - 1; // +1 offset standard
                    word |= (z << (b * bits));
                }
                packed[g * (outF / packFactor) + pj] = word;
            }
        }
        return packed;
    }

    @Override
    public String getName() {
        return "GPTQ";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return false; // Deliberately unavailable until validated against a reference implementation.
    }
}
