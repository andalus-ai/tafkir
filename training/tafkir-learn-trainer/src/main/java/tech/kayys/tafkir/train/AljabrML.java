package tech.kayys.tafkir.ml;

/**
 * @deprecated Use {@link Alkhawarizm} as the unified ML entry point.
 *             {@code AlkhawarizmML} remains as a thin source-compatible shim for
 *             existing code while all implementation lives in {@link Alkhawarizm}.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public class AlkhawarizmML {
    public static class DL extends Alkhawarizm.DL {
    }

    public static class ML extends Alkhawarizm.ML {
    }

    public static class Selection extends Alkhawarizm.Selection {
    }

    public static class Hub extends Alkhawarizm.Hub {
    }

    public static class Export extends Alkhawarizm.Export {
    }
}
