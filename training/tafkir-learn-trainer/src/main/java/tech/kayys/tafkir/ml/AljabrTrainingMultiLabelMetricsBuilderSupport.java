package tech.kayys.tafkir.ml;

/**
 * Fluent multi-label metric registration for training option builders.
 */
public abstract class AlkhawarizmTrainingMultiLabelMetricsBuilderSupport<
        B extends AlkhawarizmTrainingMultiLabelMetricsBuilderSupport<B>>
        extends AlkhawarizmTrainingBinaryMetricsBuilderSupport<B> {

    public B multiLabelExactMatchMetric() {
        return metric(Alkhawarizm.DL.multiLabelExactMatchMetric());
    }

    public B multiLabelExactMatchMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelExactMatchMetric(logitThreshold));
    }

    public B multiLabelHammingLossMetric() {
        return metric(Alkhawarizm.DL.multiLabelHammingLossMetric());
    }

    public B multiLabelHammingLossMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelHammingLossMetric(logitThreshold));
    }

    public B multiLabelConfusionMatrixMetric() {
        return metric(Alkhawarizm.DL.multiLabelConfusionMatrixMetric());
    }

    public B multiLabelConfusionMatrixMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelConfusionMatrixMetric(logitThreshold));
    }

    public B multiLabelMicroPrecisionMetric() {
        return metric(Alkhawarizm.DL.multiLabelMicroPrecisionMetric());
    }

    public B multiLabelMicroPrecisionMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelMicroPrecisionMetric(logitThreshold));
    }

    public B multiLabelMicroRecallMetric() {
        return metric(Alkhawarizm.DL.multiLabelMicroRecallMetric());
    }

    public B multiLabelMicroRecallMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelMicroRecallMetric(logitThreshold));
    }

    public B multiLabelMicroF1Metric() {
        return metric(Alkhawarizm.DL.multiLabelMicroF1Metric());
    }

    public B multiLabelMicroF1Metric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelMicroF1Metric(logitThreshold));
    }

    public B multiLabelSamplePrecisionMetric() {
        return metric(Alkhawarizm.DL.multiLabelSamplePrecisionMetric());
    }

    public B multiLabelSamplePrecisionMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelSamplePrecisionMetric(logitThreshold));
    }

    public B multiLabelSampleRecallMetric() {
        return metric(Alkhawarizm.DL.multiLabelSampleRecallMetric());
    }

    public B multiLabelSampleRecallMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelSampleRecallMetric(logitThreshold));
    }

    public B multiLabelSampleF1Metric() {
        return metric(Alkhawarizm.DL.multiLabelSampleF1Metric());
    }

    public B multiLabelSampleF1Metric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelSampleF1Metric(logitThreshold));
    }

    public B multiLabelSampleJaccardMetric() {
        return metric(Alkhawarizm.DL.multiLabelSampleJaccardMetric());
    }

    public B multiLabelSampleJaccardMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelSampleJaccardMetric(logitThreshold));
    }

    public B multiLabelMacroPrecisionMetric() {
        return metric(Alkhawarizm.DL.multiLabelMacroPrecisionMetric());
    }

    public B multiLabelMacroPrecisionMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelMacroPrecisionMetric(logitThreshold));
    }

    public B multiLabelMacroRecallMetric() {
        return metric(Alkhawarizm.DL.multiLabelMacroRecallMetric());
    }

    public B multiLabelMacroRecallMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelMacroRecallMetric(logitThreshold));
    }

    public B multiLabelMacroF1Metric() {
        return metric(Alkhawarizm.DL.multiLabelMacroF1Metric());
    }

    public B multiLabelMacroF1Metric(float logitThreshold) {
        return metric(Alkhawarizm.DL.multiLabelMacroF1Metric(logitThreshold));
    }

    public B multiLabelMacroRocAucMetric() {
        return metric(Alkhawarizm.DL.multiLabelMacroRocAucMetric());
    }

    public B multiLabelMacroAurocMetric() {
        return metric(Alkhawarizm.DL.multiLabelMacroAurocMetric());
    }

    public B multiLabelMacroAveragePrecisionMetric() {
        return metric(Alkhawarizm.DL.multiLabelMacroAveragePrecisionMetric());
    }

    public B multiLabelLabelRankingAveragePrecisionMetric() {
        return metric(Alkhawarizm.DL.multiLabelLabelRankingAveragePrecisionMetric());
    }

    public B multiLabelLrapMetric() {
        return metric(Alkhawarizm.DL.multiLabelLrapMetric());
    }

    public B multiLabelRankingLossMetric() {
        return metric(Alkhawarizm.DL.multiLabelRankingLossMetric());
    }

    public B multiLabelCoverageErrorMetric() {
        return metric(Alkhawarizm.DL.multiLabelCoverageErrorMetric());
    }

    public B multiLabelMacroBestF1Metric() {
        return metric(Alkhawarizm.DL.multiLabelMacroBestF1Metric());
    }

    public B multiLabelBestF1ThresholdsMetric() {
        return metric(Alkhawarizm.DL.multiLabelBestF1ThresholdsMetric());
    }

    public B multiLabelRankingMetrics() {
        return multiLabelMacroRocAucMetric()
                .multiLabelMacroAveragePrecisionMetric()
                .multiLabelLabelRankingAveragePrecisionMetric()
                .multiLabelRankingLossMetric()
                .multiLabelCoverageErrorMetric();
    }

    public B multiLabelThresholdTuningMetrics() {
        return multiLabelBestF1ThresholdsMetric();
    }

    public B multiLabelMicroMetrics() {
        return multiLabelMicroPrecisionMetric()
                .multiLabelMicroRecallMetric()
                .multiLabelMicroF1Metric();
    }

    public B multiLabelMicroMetrics(float logitThreshold) {
        return multiLabelMicroPrecisionMetric(logitThreshold)
                .multiLabelMicroRecallMetric(logitThreshold)
                .multiLabelMicroF1Metric(logitThreshold);
    }

    public B multiLabelSampleMetrics() {
        return multiLabelSamplePrecisionMetric()
                .multiLabelSampleRecallMetric()
                .multiLabelSampleF1Metric()
                .multiLabelSampleJaccardMetric();
    }

    public B multiLabelSampleMetrics(float logitThreshold) {
        return multiLabelSamplePrecisionMetric(logitThreshold)
                .multiLabelSampleRecallMetric(logitThreshold)
                .multiLabelSampleF1Metric(logitThreshold)
                .multiLabelSampleJaccardMetric(logitThreshold);
    }

    public B multiLabelBinaryMetrics() {
        return multiLabelExactMatchMetric()
                .multiLabelHammingLossMetric()
                .multiLabelMicroMetrics()
                .multiLabelSampleMetrics()
                .multiLabelMacroPrecisionMetric()
                .multiLabelMacroRecallMetric()
                .multiLabelMacroF1Metric();
    }

    public B multiLabelBinaryMetrics(float logitThreshold) {
        return multiLabelExactMatchMetric(logitThreshold)
                .multiLabelHammingLossMetric(logitThreshold)
                .multiLabelMicroMetrics(logitThreshold)
                .multiLabelSampleMetrics(logitThreshold)
                .multiLabelMacroPrecisionMetric(logitThreshold)
                .multiLabelMacroRecallMetric(logitThreshold)
                .multiLabelMacroF1Metric(logitThreshold);
    }

    public B multiLabelMetrics() {
        return multiLabelBinaryMetrics();
    }

    public B multiLabelMetrics(float logitThreshold) {
        return multiLabelBinaryMetrics(logitThreshold);
    }
}
