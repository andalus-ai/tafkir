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
        int outF = (int) quantizedTensor.size(0);
        int inF = (int) quantizedTensor.size(1);
        int bits = config.getBits();
        int groupSize = quantizedTensor.groupSize();
        
        AutoRoundConfig arConfig = new AutoRoundConfig(bits, groupSize, true, AutoRoundConfig.ScaleDtype.FLOAT32, AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, "exllamav2", null);
        AutoRoundDequantizer dequantizer = new AutoRoundDequantizer(arConfig);
        
        MemorySegment dataSeg = quantizedTensor.dataSegment();
        int[] qweightInts = new int[(int) (dataSeg.byteSize() / 4)];
        for (int i = 0; i < qweightInts.length; i++) {
            qweightInts[i] = dataSeg.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), (long) i * 4);
        }
        
        float[] scales = quantizedTensor.scales().toArray(ValueLayout.JAVA_FLOAT);
        int[] zp = quantizedTensor.zeros().toArray(ValueLayout.JAVA_INT);
        
        float[] output = new float[outF * inF];
        dequantizer.dequantize(qweightInts, scales, zp, inF, outF, output);
        
        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
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
