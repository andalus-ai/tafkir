package tech.kayys.tafkir.ml;

/**
 * Fluent binary classification metric registration for training option builders.
 */
public abstract class AlkhawarizmTrainingBinaryMetricsBuilderSupport<
        B extends AlkhawarizmTrainingBinaryMetricsBuilderSupport<B>>
        extends AlkhawarizmTrainingClassificationMetricsBuilderSupport<B> {

    public B binaryAccuracyMetric() {
        return metric(Alkhawarizm.DL.binaryAccuracyMetric());
    }

    public B binaryAccuracyMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryAccuracyMetric(logitThreshold));
    }

    public B binaryBalancedAccuracyMetric() {
        return metric(Alkhawarizm.DL.binaryBalancedAccuracyMetric());
    }

    public B binaryBalancedAccuracyMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryBalancedAccuracyMetric(logitThreshold));
    }

    public B binaryMatthewsCorrelationCoefficientMetric() {
        return metric(Alkhawarizm.DL.binaryMatthewsCorrelationCoefficientMetric());
    }

    public B binaryMatthewsCorrelationCoefficientMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryMatthewsCorrelationCoefficientMetric(logitThreshold));
    }

    public B binaryMccMetric() {
        return metric(Alkhawarizm.DL.binaryMccMetric());
    }

    public B binaryMccMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryMccMetric(logitThreshold));
    }

    public B binaryCohensKappaMetric() {
        return metric(Alkhawarizm.DL.binaryCohensKappaMetric());
    }

    public B binaryCohensKappaMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryCohensKappaMetric(logitThreshold));
    }

    public B binaryKappaMetric() {
        return metric(Alkhawarizm.DL.binaryKappaMetric());
    }

    public B binaryKappaMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryKappaMetric(logitThreshold));
    }

    public B binaryConfusionMatrixMetric() {
        return metric(Alkhawarizm.DL.binaryConfusionMatrixMetric());
    }

    public B binaryConfusionMatrixMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryConfusionMatrixMetric(logitThreshold));
    }

    public B binaryPrecisionMetric() {
        return metric(Alkhawarizm.DL.binaryPrecisionMetric());
    }

    public B binaryPrecisionMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryPrecisionMetric(logitThreshold));
    }

    public B binaryRecallMetric() {
        return metric(Alkhawarizm.DL.binaryRecallMetric());
    }

    public B binaryRecallMetric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryRecallMetric(logitThreshold));
    }

    public B binaryF1Metric() {
        return metric(Alkhawarizm.DL.binaryF1Metric());
    }

    public B binaryF1Metric(float logitThreshold) {
        return metric(Alkhawarizm.DL.binaryF1Metric(logitThreshold));
    }

    public B binaryRocAucMetric() {
        return metric(Alkhawarizm.DL.binaryRocAucMetric());
    }

    public B binaryAurocMetric() {
        return metric(Alkhawarizm.DL.binaryAurocMetric());
    }

    public B binaryAveragePrecisionMetric() {
        return metric(Alkhawarizm.DL.binaryAveragePrecisionMetric());
    }

    public B binaryBestF1Metric() {
        return metric(Alkhawarizm.DL.binaryBestF1Metric());
    }

    public B binaryBestF1ThresholdMetric() {
        return metric(Alkhawarizm.DL.binaryBestF1ThresholdMetric());
    }

    public B binaryPrecisionAtRecallMetric(double minimumRecall) {
        return metric(Alkhawarizm.DL.binaryPrecisionAtRecallMetric(minimumRecall));
    }

    public B binaryRecallAtPrecisionMetric(double minimumPrecision) {
        return metric(Alkhawarizm.DL.binaryRecallAtPrecisionMetric(minimumPrecision));
    }

    public B binaryBrierScoreMetric() {
        return metric(Alkhawarizm.DL.binaryBrierScoreMetric());
    }

    public B binaryLogLossMetric() {
        return metric(Alkhawarizm.DL.binaryLogLossMetric());
    }

    public B binaryCrossEntropyMetric() {
        return metric(Alkhawarizm.DL.binaryCrossEntropyMetric());
    }

    public B binaryExpectedCalibrationErrorMetric() {
        return metric(Alkhawarizm.DL.binaryExpectedCalibrationErrorMetric());
    }

    public B binaryExpectedCalibrationErrorMetric(int bins) {
        return metric(Alkhawarizm.DL.binaryExpectedCalibrationErrorMetric(bins));
    }

    public B binaryRankingMetrics() {
        return binaryRocAucMetric()
                .binaryAveragePrecisionMetric();
    }

    public B binaryImbalanceMetrics() {
        return binaryBalancedAccuracyMetric()
                .binaryMccMetric();
    }

    public B binaryImbalanceMetrics(float logitThreshold) {
        return binaryBalancedAccuracyMetric(logitThreshold)
                .binaryMccMetric(logitThreshold);
    }

    public B binaryAgreementMetrics() {
        return binaryKappaMetric();
    }

    public B binaryAgreementMetrics(float logitThreshold) {
        return binaryKappaMetric(logitThreshold);
    }

    public B binaryThresholdTuningMetrics() {
        return binaryBestF1ThresholdMetric();
    }

    public B binaryThresholdTuningMetrics(double minimumRecall, double minimumPrecision) {
        return binaryBestF1ThresholdMetric()
                .binaryPrecisionAtRecallMetric(minimumRecall)
                .binaryRecallAtPrecisionMetric(minimumPrecision);
    }

    public B binaryCalibrationMetrics() {
        return binaryBrierScoreMetric()
                .binaryLogLossMetric()
                .binaryExpectedCalibrationErrorMetric();
    }

    public B binaryCalibrationMetrics(int bins) {
        return binaryBrierScoreMetric()
                .binaryLogLossMetric()
                .binaryExpectedCalibrationErrorMetric(bins);
    }

    public B binaryClassificationMetrics() {
        return binaryAccuracyMetric()
                .binaryPrecisionMetric()
                .binaryRecallMetric()
                .binaryF1Metric();
    }

    public B binaryClassificationMetrics(float logitThreshold) {
        return binaryAccuracyMetric(logitThreshold)
                .binaryPrecisionMetric(logitThreshold)
                .binaryRecallMetric(logitThreshold)
                .binaryF1Metric(logitThreshold);
    }

    public B binaryMetrics() {
        return binaryClassificationMetrics();
    }

    public B binaryMetrics(float logitThreshold) {
        return binaryClassificationMetrics(logitThreshold);
    }
}
