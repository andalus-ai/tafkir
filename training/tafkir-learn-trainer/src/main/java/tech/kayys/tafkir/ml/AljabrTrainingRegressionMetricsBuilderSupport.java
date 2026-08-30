package tech.kayys.tafkir.ml;

/**
 * Fluent regression metric registration for training option builders.
 */
public abstract class AlkhawarizmTrainingRegressionMetricsBuilderSupport<
        B extends AlkhawarizmTrainingRegressionMetricsBuilderSupport<B>>
        extends AlkhawarizmTrainingMultiLabelMetricsBuilderSupport<B> {

    public B meanAbsoluteErrorMetric() {
        return metric(Alkhawarizm.DL.meanAbsoluteErrorMetric());
    }

    public B maeMetric() {
        return meanAbsoluteErrorMetric();
    }

    public B meanSquaredErrorMetric() {
        return metric(Alkhawarizm.DL.meanSquaredErrorMetric());
    }

    public B mseMetric() {
        return meanSquaredErrorMetric();
    }

    public B rootMeanSquaredErrorMetric() {
        return metric(Alkhawarizm.DL.rootMeanSquaredErrorMetric());
    }

    public B rmseMetric() {
        return rootMeanSquaredErrorMetric();
    }

    public B meanSquaredLogErrorMetric() {
        return metric(Alkhawarizm.DL.meanSquaredLogErrorMetric());
    }

    public B msleMetric() {
        return meanSquaredLogErrorMetric();
    }

    public B rootMeanSquaredLogErrorMetric() {
        return metric(Alkhawarizm.DL.rootMeanSquaredLogErrorMetric());
    }

    public B rmsleMetric() {
        return rootMeanSquaredLogErrorMetric();
    }

    public B meanPoissonDevianceMetric() {
        return metric(Alkhawarizm.DL.meanPoissonDevianceMetric());
    }

    public B meanPoissonDevianceMetric(boolean logInput) {
        return metric(Alkhawarizm.DL.meanPoissonDevianceMetric(logInput));
    }

    public B meanPoissonDevianceMetric(boolean logInput, double eps) {
        return metric(Alkhawarizm.DL.meanPoissonDevianceMetric(logInput, eps));
    }

    public B poissonDevianceMetric() {
        return meanPoissonDevianceMetric();
    }

    public B poissonLogRateDevianceMetric() {
        return metric(Alkhawarizm.DL.poissonLogRateDevianceMetric());
    }

    public B meanTweedieDevianceMetric() {
        return metric(Alkhawarizm.DL.meanTweedieDevianceMetric());
    }

    public B meanTweedieDevianceMetric(double power) {
        return metric(Alkhawarizm.DL.meanTweedieDevianceMetric(power));
    }

    public B meanTweedieDevianceMetric(double power, boolean logInput) {
        return metric(Alkhawarizm.DL.meanTweedieDevianceMetric(power, logInput));
    }

    public B meanTweedieDevianceMetric(double power, boolean logInput, double eps) {
        return metric(Alkhawarizm.DL.meanTweedieDevianceMetric(power, logInput, eps));
    }

    public B tweedieDevianceMetric() {
        return meanTweedieDevianceMetric();
    }

    public B compoundPoissonGammaDevianceMetric() {
        return metric(Alkhawarizm.DL.compoundPoissonGammaDevianceMetric());
    }

    public B medianAbsoluteErrorMetric() {
        return metric(Alkhawarizm.DL.medianAbsoluteErrorMetric());
    }

    public B medaeMetric() {
        return medianAbsoluteErrorMetric();
    }

    public B maxErrorMetric() {
        return metric(Alkhawarizm.DL.maxErrorMetric());
    }

    public B pinballLossMetric(double quantile) {
        return metric(Alkhawarizm.DL.pinballLossMetric(quantile));
    }

    public B meanPinballLossMetric(double quantile) {
        return metric(Alkhawarizm.DL.meanPinballLossMetric(quantile));
    }

    public B predictionIntervalCoverageMetric() {
        return metric(Alkhawarizm.DL.predictionIntervalCoverageMetric());
    }

    public B picpMetric() {
        return predictionIntervalCoverageMetric();
    }

    public B predictionIntervalMeanWidthMetric() {
        return metric(Alkhawarizm.DL.predictionIntervalMeanWidthMetric());
    }

    public B predictionIntervalNormalizedMeanWidthMetric() {
        return metric(Alkhawarizm.DL.predictionIntervalNormalizedMeanWidthMetric());
    }

    public B r2ScoreMetric() {
        return metric(Alkhawarizm.DL.r2ScoreMetric());
    }

    public B r2Metric() {
        return r2ScoreMetric();
    }

    public B meanAbsolutePercentageErrorMetric() {
        return metric(Alkhawarizm.DL.meanAbsolutePercentageErrorMetric());
    }

    public B mapeMetric() {
        return meanAbsolutePercentageErrorMetric();
    }

    public B symmetricMeanAbsolutePercentageErrorMetric() {
        return metric(Alkhawarizm.DL.symmetricMeanAbsolutePercentageErrorMetric());
    }

    public B smapeMetric() {
        return symmetricMeanAbsolutePercentageErrorMetric();
    }

    public B meanBiasErrorMetric() {
        return metric(Alkhawarizm.DL.meanBiasErrorMetric());
    }

    public B mbeMetric() {
        return meanBiasErrorMetric();
    }

    public B explainedVarianceMetric() {
        return metric(Alkhawarizm.DL.explainedVarianceMetric());
    }

    public B explainedVarianceScoreMetric() {
        return explainedVarianceMetric();
    }

    public B regressionMetrics() {
        return meanAbsoluteErrorMetric()
                .meanSquaredErrorMetric()
                .rootMeanSquaredErrorMetric()
                .r2ScoreMetric();
    }

    public B regressionExtendedMetrics() {
        return regressionMetrics()
                .medianAbsoluteErrorMetric()
                .maxErrorMetric()
                .meanAbsolutePercentageErrorMetric()
                .symmetricMeanAbsolutePercentageErrorMetric()
                .meanBiasErrorMetric()
                .explainedVarianceMetric();
    }

    public B regressionLogScaleMetrics() {
        return meanSquaredLogErrorMetric()
                .rootMeanSquaredLogErrorMetric();
    }

    public B countRegressionMetrics() {
        return poissonLogRateDevianceMetric()
                .compoundPoissonGammaDevianceMetric();
    }

    public B regressionQuantileMetrics() {
        return pinballLossMetric(0.1)
                .pinballLossMetric(0.5)
                .pinballLossMetric(0.9);
    }

    public B regressionPredictionIntervalMetrics() {
        return predictionIntervalCoverageMetric()
                .predictionIntervalMeanWidthMetric()
                .predictionIntervalNormalizedMeanWidthMetric();
    }
}
