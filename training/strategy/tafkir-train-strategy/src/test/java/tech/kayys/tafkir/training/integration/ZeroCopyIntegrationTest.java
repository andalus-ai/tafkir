package tech.kayys.tafkir.training.integration;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.memory.BorrowableMemoryStore;
import tech.kayys.aljabr.core.memory.MemoryStoreLease;
import tech.kayys.aljabr.core.memory.helixdb.HelixDbMemoryStore;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that Tafkir can consume Aljabr-owned memory through the scoped
 * borrowing contract without depending on a persistent database.
 */
class ZeroCopyIntegrationTest {

    @Test
    void borrowsAljabrMemoryForTrainingWithoutCopyingTheReadView() {
        byte[] contextKey = "agent_context_1".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] modelKey = "agent_adapter_1".getBytes(java.nio.charset.StandardCharsets.UTF_8);

        try (BorrowableMemoryStore store = new HelixDbMemoryStore("unused-test-path")) {
            // The provider copies input into its own arena. Closing this source
            // arena proves that subsequent reads do not depend on caller memory.
            try (Arena sourceArena = Arena.ofConfined()) {
                MemorySegment context = sourceArena.allocate(1024);
                MemorySegment adapter = sourceArena.allocate(1024);
                context.set(ValueLayout.JAVA_BYTE, 0, (byte) 0x2A);
                adapter.set(ValueLayout.JAVA_BYTE, 0, (byte) 0x5A);
                store.put(contextKey, context);
                store.put(modelKey, adapter);
            }

            try (MemoryStoreLease contextLease = store.borrow(contextKey).orElseThrow();
                 MemoryStoreLease adapterLease = store.borrow(modelKey).orElseThrow()) {
                MemorySegment context = contextLease.segment();
                MemorySegment adapter = adapterLease.segment();

                assertEquals(1024, context.byteSize());
                assertEquals(1024, adapter.byteSize());
                assertEquals((byte) 0x2A, context.get(ValueLayout.JAVA_BYTE, 0));
                assertEquals((byte) 0x5A, adapter.get(ValueLayout.JAVA_BYTE, 0));
                assertFalse(contextLease.isClosed());
                assertFalse(adapterLease.isClosed());
            }

            assertTrue(store.borrow(contextKey).isPresent(),
                    "Closing a lease releases the borrow but does not delete the stored value");
        }
    }
}
