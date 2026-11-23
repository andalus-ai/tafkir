package tech.kayys.tafkir.training.integration;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.memory.MemoryStoreLease;
import tech.kayys.aljabr.core.memory.helixdb.HelixDbMemoryStore;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies the Aljabr-to-Tafkir memory handoff contract without requiring a
 * native LibTorch installation. Native execution belongs in a separately
 * provisioned integration-test profile.
 */
class ZeroCopyIntegrationTest {

    @Test
    void borrowedTrainingInputsRemainValidUntilTheTrainingScopeEnds() {
        try (HelixDbMemoryStore store = new HelixDbMemoryStore("unused-in-memory-path");
                Arena inputArena = Arena.ofConfined()) {
            byte[] contextKey = "agent_context_1".getBytes();
            byte[] adapterKey = "agent_adapter_1".getBytes();

            MemorySegment contextInput = inputArena.allocate(1024);
            MemorySegment adapterInput = inputArena.allocate(1024);
            contextInput.set(ValueLayout.JAVA_FLOAT, 0, 0.25f);
            adapterInput.set(ValueLayout.JAVA_FLOAT, 0, 0.75f);
            store.put(contextKey, contextInput);
            store.put(adapterKey, adapterInput);

            // Keep both provider-owned borrows alive for the complete native
            // training operation. A real FFM call receives lease.segment().
            try (MemoryStoreLease contextLease = store.borrow(contextKey).orElseThrow();
                    MemoryStoreLease adapterLease = store.borrow(adapterKey).orElseThrow()) {
                MemorySegment contextSegment = contextLease.segment();
                MemorySegment adapterSegment = adapterLease.segment();

                assertEquals(1024, contextSegment.byteSize());
                assertEquals(1024, adapterSegment.byteSize());
                assertEquals(0.25f, contextSegment.get(ValueLayout.JAVA_FLOAT, 0));
                assertEquals(0.75f, adapterSegment.get(ValueLayout.JAVA_FLOAT, 0));

                // The FFM consumer must complete before these leases close.
                // The segment aliases provider storage; no readback copy occurs.
            }

            // Closed leases no longer expose the borrowed segment.
            var closedLease = store.borrow(contextKey).orElseThrow();
            closedLease.close();
            assertThrows(IllegalStateException.class, closedLease::segment);
        }
    }
}
