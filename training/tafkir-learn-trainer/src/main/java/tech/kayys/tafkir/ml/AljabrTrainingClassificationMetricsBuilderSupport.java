package tech.kayys.tafkir.ml;

/**
 * Fluent multi-class classification metric registration for training option builders.
 */
public abstract class AlkhawarizmTrainingClassificationMetricsBuilderSupport<
        B extends AlkhawarizmTrainingClassificationMetricsBuilderSupport<B>>
        extends AlkhawarizmTrainingMetricRegistrySupport<B> {

    public B accuracyMetric() {
        return metric(Alkhawarizm.DL.accuracyMetric());
    }

    public B confusionMatrixMetric() {
        return metric(Alkhawarizm.DL.confusionMatrixMetric());
    }

    public B classificationConfusionMatrixMetric() {
        return metric(Alkhawarizm.DL.classificationConfusionMatrixMetric());
    }

    public B topKAccuracyMetric(int k) {
        return metric(Alkhawarizm.DL.topKAccuracyMetric(k));
    }

    public B classificationLogLossMetric() {
        return metric(Alkhawarizm.DL.classificationLogLossMetric());
    }

    public B classificationCrossEntropyMetric() {
        return metric(Alkhawarizm.DL.classificationCrossEntropyMetric());
    }

    public B classificationBalancedAccuracyMetric() {
        return metric(Alkhawarizm.DL.classificationBalancedAccuracyMetric());
    }

    public B balancedAccuracyMetric() {
        return metric(Alkhawarizm.DL.balancedAccuracyMetric());
    }

    public B classificationMatthewsCorrelationCoefficientMetric() {
        return metric(Alkhawarizm.DL.classificationMatthewsCorrelationCoefficientMetric());
    }

    public B classificationMccMetric() {
        return metric(Alkhawarizm.DL.classificationMccMetric());
    }

    public B matthewsCorrelationCoefficientMetric() {
        return metric(Alkhawarizm.DL.matthewsCorrelationCoefficientMetric());
    }

    public B mccMetric() {
        return metric(Alkhawarizm.DL.mccMetric());
    }

    public B classificationWeightedPrecisionMetric() {
        return metric(Alkhawarizm.DL.classificationWeightedPrecisionMetric());
    }

    public B classificationWeightedRecallMetric() {
        return metric(Alkhawarizm.DL.classificationWeightedRecallMetric());
    }

    public B classificationWeightedF1Metric() {
        return metric(Alkhawarizm.DL.classificationWeightedF1Metric());
    }

    public B classificationCohensKappaMetric() {
        return metric(Alkhawarizm.DL.classificationCohensKappaMetric());
    }

    public B classificationKappaMetric() {
        return metric(Alkhawarizm.DL.classificationKappaMetric());
    }

    public B cohensKappaMetric() {
        return metric(Alkhawarizm.DL.cohensKappaMetric());
    }

    public B kappaMetric() {
        return metric(Alkhawarizm.DL.kappaMetric());
    }

    public B precisionMetric() {
        return metric(Alkhawarizm.DL.precisionMetric());
    }

    public B recallMetric() {
        return metric(Alkhawarizm.DL.recallMetric());
    }

    public B f1Metric() {
        return metric(Alkhawarizm.DL.f1Metric());
    }

    public B macroF1Metric() {
        return metric(Alkhawarizm.DL.macroF1Metric());
    }

    public B classificationMetrics() {
        return accuracyMetric()
                .classificationLogLossMetric()
                .precisionMetric()
                .recallMetric()
                .f1Metric();
    }

    public B classificationMacroRocAucMetric() {
        return metric(Alkhawarizm.DL.classificationMacroRocAucMetric());
    }

    public B classificationMacroAurocMetric() {
        return metric(Alkhawarizm.DL.classificationMacroAurocMetric());
    }

    public B classificationMacroAveragePrecisionMetric() {
        return metric(Alkhawarizm.DL.classificationMacroAveragePrecisionMetric());
    }

    public B classificationRankingMetrics() {
        return classificationMacroRocAucMetric()
                .classificationMacroAveragePrecisionMetric();
    }

    public B classificationImbalanceMetrics() {
        return classificationBalancedAccuracyMetric()
                .classificationMccMetric()
                .classificationWeightedPrecisionMetric()
                .classificationWeightedRecallMetric()
                .classificationWeightedF1Metric();
    }

    public B classificationAgreementMetrics() {
        return classificationKappaMetric();
    }

    public B classificationBrierScoreMetric() {
        return metric(Alkhawarizm.DL.classificationBrierScoreMetric());
    }

    public B classificationExpectedCalibrationErrorMetric() {
        return metric(Alkhawarizm.DL.classificationExpectedCalibrationErrorMetric());
    }

    public B classificationExpectedCalibrationErrorMetric(int bins) {
        return metric(Alkhawarizm.DL.classificationExpectedCalibrationErrorMetric(bins));
    }

    public B classificationCalibrationMetrics() {
        return classificationBrierScoreMetric()
                .classificationExpectedCalibrationErrorMetric();
    }

    public B classificationCalibrationMetrics(int bins) {
        return classificationBrierScoreMetric()
                .classificationExpectedCalibrationErrorMetric(bins);
    }
}
