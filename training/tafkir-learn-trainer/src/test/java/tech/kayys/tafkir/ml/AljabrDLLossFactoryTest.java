package tech.kayys.tafkir.ml;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import org.junit.jupiter.api.Test;
import tech.kayys.tafkir.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.tafkir.ml.nn.loss.BinaryFocalWithLogitsLoss;
import tech.kayys.tafkir.ml.nn.loss.CausalLanguageModelingLoss;
import tech.kayys.tafkir.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.tafkir.ml.nn.loss.FocalLoss;
import tech.kayys.tafkir.ml.nn.loss.GaussianNllLoss;
import tech.kayys.tafkir.ml.nn.loss.HuberLoss;
import tech.kayys.tafkir.ml.nn.loss.L1Loss;
import tech.kayys.tafkir.ml.nn.loss.MSELoss;
import tech.kayys.tafkir.ml.nn.loss.NegativeBinomialNllLoss;
import tech.kayys.tafkir.ml.nn.loss.PinballLoss;
import tech.kayys.tafkir.ml.nn.loss.PoissonNllLoss;
import tech.kayys.tafkir.ml.nn.loss.PredictionIntervalLoss;
import tech.kayys.tafkir.ml.nn.loss.SmoothL1Loss;
import tech.kayys.tafkir.ml.nn.loss.TweedieNllLoss;
import tech.kayys.tafkir.ml.nn.loss.ZeroInflatedNegativeBinomialNllLoss;
import tech.kayys.tafkir.ml.nn.loss.ZeroInflatedPoissonNllLoss;

class AlkhawarizmDLLossFactoryTest {

    @Test
    void exposesClassificationLossFactoriesOnAlkhawarizmFacade() {
        assertInstanceOf(CrossEntropyLoss.class, Alkhawarizm.DL.crossEntropy());
        assertInstanceOf(CrossEntropyLoss.class, Alkhawarizm.DL.crossEntropy(new float[] {1.0f, 2.0f}));
        assertInstanceOf(CausalLanguageModelingLoss.class, Alkhawarizm.DL.causalLanguageModelingLoss());
        assertInstanceOf(CausalLanguageModelingLoss.class, Alkhawarizm.DL.causalLanguageModelingLoss(-1.0f));
        assertInstanceOf(FocalLoss.class, Alkhawarizm.DL.focalLoss());
        assertInstanceOf(FocalLoss.class, Alkhawarizm.DL.focalLoss(2.0f));
        assertInstanceOf(FocalLoss.class, Alkhawarizm.DL.focalLoss(2.0f, 0.25f));
        assertInstanceOf(FocalLoss.class, Alkhawarizm.DL.focalLoss(2.0f, new float[] {1.0f, 2.0f}));
    }

    @Test
    void exposesRegressionLossFactoriesOnAlkhawarizmFacade() {
        assertInstanceOf(MSELoss.class, Alkhawarizm.DL.mseLoss());
        assertInstanceOf(L1Loss.class, Alkhawarizm.DL.l1Loss());
        assertInstanceOf(HuberLoss.class, Alkhawarizm.DL.huberLoss());
        assertInstanceOf(HuberLoss.class, Alkhawarizm.DL.huberLoss(1.5f));
        assertInstanceOf(SmoothL1Loss.class, Alkhawarizm.DL.smoothL1Loss());
        assertInstanceOf(SmoothL1Loss.class, Alkhawarizm.DL.smoothL1Loss(0.75f));
        assertInstanceOf(PinballLoss.class, Alkhawarizm.DL.pinballLoss());
        assertInstanceOf(PinballLoss.class, Alkhawarizm.DL.quantileLoss(0.9));
        assertInstanceOf(PinballLoss.class, Alkhawarizm.DL.quantileLoss(0.1, 0.5, 0.9));
        assertInstanceOf(PinballLoss.class, Alkhawarizm.DL.predictionIntervalLoss(0.1, 0.9));
        assertInstanceOf(PredictionIntervalLoss.class, Alkhawarizm.DL.intervalScoreLoss());
        assertInstanceOf(PredictionIntervalLoss.class, Alkhawarizm.DL.winklerLoss(0.2, 12.0));
        assertInstanceOf(GaussianNllLoss.class, Alkhawarizm.DL.gaussianNllLoss());
        assertInstanceOf(GaussianNllLoss.class, Alkhawarizm.DL.heteroscedasticGaussianLoss(true));
    }

    @Test
    void exposesCountAndZeroInflatedLossFactoriesOnAlkhawarizmFacade() {
        assertInstanceOf(PoissonNllLoss.class, Alkhawarizm.DL.poissonNllLoss());
        assertInstanceOf(PoissonNllLoss.class, Alkhawarizm.DL.poissonNllLoss(true, true, 1e-8));
        assertInstanceOf(PoissonNllLoss.class, Alkhawarizm.DL.countNllLoss());
        assertInstanceOf(TweedieNllLoss.class, Alkhawarizm.DL.tweedieNllLoss());
        assertInstanceOf(TweedieNllLoss.class, Alkhawarizm.DL.compoundPoissonGammaLoss(1.5));
        assertInstanceOf(NegativeBinomialNllLoss.class, Alkhawarizm.DL.negativeBinomialNllLoss());
        assertInstanceOf(NegativeBinomialNllLoss.class, Alkhawarizm.DL.overdispersedCountNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Alkhawarizm.DL.zeroInflatedPoissonNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Alkhawarizm.DL.zipNllLoss());
        assertInstanceOf(ZeroInflatedPoissonNllLoss.class, Alkhawarizm.DL.excessZeroCountNllLoss());
        assertInstanceOf(ZeroInflatedNegativeBinomialNllLoss.class, Alkhawarizm.DL.zeroInflatedNegativeBinomialNllLoss());
        assertInstanceOf(ZeroInflatedNegativeBinomialNllLoss.class, Alkhawarizm.DL.zinbNllLoss());
        assertInstanceOf(
                ZeroInflatedNegativeBinomialNllLoss.class,
                Alkhawarizm.DL.excessZeroOverdispersedCountNllLoss());
    }

    @Test
    void exposesBinaryLogitLossFactoriesOnAlkhawarizmFacade() {
        assertInstanceOf(BCEWithLogitsLoss.class, Alkhawarizm.DL.bceWithLogitsLoss());
        assertInstanceOf(BCEWithLogitsLoss.class, Alkhawarizm.DL.bceWithLogitsLoss(3.0f));
        assertInstanceOf(BCEWithLogitsLoss.class, Alkhawarizm.DL.bceWithLogitsLoss(new float[] {2.0f, 3.0f}));
        assertInstanceOf(BCEWithLogitsLoss.class, Alkhawarizm.DL.binaryCrossEntropyWithLogits());
        assertInstanceOf(BCEWithLogitsLoss.class, Alkhawarizm.DL.binaryCrossEntropyWithLogits(2.0f));
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Alkhawarizm.DL.binaryFocalWithLogitsLoss());
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Alkhawarizm.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f));
        assertInstanceOf(BinaryFocalWithLogitsLoss.class, Alkhawarizm.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f, 3.0f));
        assertInstanceOf(
                BinaryFocalWithLogitsLoss.class,
                Alkhawarizm.DL.binaryFocalWithLogitsLoss(2.0f, 0.25f, new float[] {2.0f, 3.0f}));
    }
}
