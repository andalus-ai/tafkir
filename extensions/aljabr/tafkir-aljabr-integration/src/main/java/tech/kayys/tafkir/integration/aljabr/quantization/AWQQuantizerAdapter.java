package tech.kayys.tafkir.integration.aljabr.quantization;

import tech.kayys.aljabr.safetensor.quantization.quantizer.Quantizer;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.quantization.QuantConfig;
import tech.kayys.tafkir.quantizer.awq.AWQDequantizer;
import tech.kayys.tafkir.quantizer.awq.AWQConfig;
import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Adapter for Activation-aware Weight Quantization (AWQ).
 * @author bhangun
 */
public class AWQQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(AWQQuantizerAdapter.class);

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        throw new UnsupportedOperationException("AWQ adapter is disabled until its algorithm has reference-backed numerical validation.");
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        int inF = (int) quantizedTensor.size(0);
        int outF = (int) quantizedTensor.size(1);
        int bits = config.getBits();
        int groupSize = quantizedTensor.groupSize();
        
        AWQConfig awqConfig = new AWQConfig(bits, groupSize, AWQConfig.KernelFormat.GEMM, true, "float32", false, 128, 2048, true, null);
        AWQDequantizer dequantizer = new AWQDequantizer(awqConfig);
        
        MemorySegment dataSeg = quantizedTensor.dataSegment();
        int[] qweightInts = new int[(int) (dataSeg.byteSize() / 4)];
        for (int i = 0; i < qweightInts.length; i++) {
            qweightInts[i] = dataSeg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), (long) i * 4);
        }
        
        float[] scales = quantizedTensor.scales().toArray(ValueLayout.JAVA_FLOAT);
        float[] zeros = quantizedTensor.zeros().toArray(ValueLayout.JAVA_FLOAT);
        
        // AutoAWQ uses (q - zero) * scale, core expects FP16 scales and packed zeros.
        // We'll use a simplified dequantization loop if the core dequantizer doesn't match our storage exactly.
        // But the AWQDequantizer.dequantize is SIMD optimized.
        
        int numGroups = (inF + groupSize - 1) / groupSize;
        int packFactor = 32 / bits;
        int mask = (1 << bits) - 1;

        float[] output = new float[inF * outF];

        for (int g = 0; g < numGroups; g++) {
            int iStart = g * groupSize;
            int iEnd = Math.min(iStart + groupSize, inF);

            for (int j = 0; j < outF; j++) {
                float scale = scales[g * outF + j];
                float zero = zeros[g * outF + j];

                for (int i = iStart; i < iEnd; i++) {
                    int pi = i / packFactor;
                    int b = i % packFactor;
                    int q = (qweightInts[pi * outF + j] >>> (b * bits)) & mask;
                    output[i * outF + j] = (q - zero) * scale;
                }
            }
        }

        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
    }

    private short floatToFp16(float v) {
        return (short) Float.floatToFloat16(v);
    }

    private int[] packZeros(float[] zeros, int numGroups, int outF, int bits) {
        int packFactor = 32 / bits;
        int[] packed = new int[numGroups * (outF / packFactor)];
        for (int g = 0; g < numGroups; g++) {
            for (int j = 0; j < outF; j++) {
                int qz = (int) zeros[g * outF + j] - 1; // AutoAWQ +1 offset
                int pj = j / packFactor;
                int b = j % packFactor;
                packed[g * (outF / packFactor) + pj] |= (qz << (b * bits));
            }
        }
        return packed;
    }

    @Override
    public String getName() {
        return "AWQ";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return false; // Deliberately unavailable until validated against a reference implementation.
    }
}
