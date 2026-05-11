# dpctl submit path analysis

1. Python/Cython entrypoint:
   - file:
   - funzione:

2. Argument preparation:
   - file:
   - funzioni:

3. Range preparation:
   - file:
   - funzioni:

4. Native C API:
   - header:
   - funzioni:

5. C++ implementation:
   - file:
   - funzioni:

6. Actual SYCL submit point:
   - file:
   - funzione:
   - riga/descrizione:


dpctl/_sycl_queue.pyx
    _populate_args        → riga 936
    _populate_range       → riga 990
    submit_async          → riga 1172
    submit                → intorno alla riga 1365

dpctl/_sycl_queue.pyx
    chiama DPCTLQueue_SubmitRange    → riga 1272
    chiama DPCTLQueue_SubmitNDRange  → riga 1310

libsyclinterface/include/syclinterface/dpctl_sycl_queue_interface.h
    dichiara DPCTLQueue_SubmitRange    → riga 219
    dichiara DPCTLQueue_SubmitNDRange  → riga 268

libsyclinterface/source/dpctl_sycl_queue_interface.cpp
    implementa DPCTLQueue_SubmitRange    → riga 490
    implementa DPCTLQueue_SubmitNDRange  → riga 554



# Non-invasive SYnergy backend integration in dpctl

## Obiettivo

L’obiettivo è integrare nella fork di `dpctl` solo la parte nativa necessaria a eseguire l’invio dei kernel tramite `synergy::queue::submit(...)`, senza spostare l’interfaccia principale del progetto dentro `dpctl`.

Il progetto principale rimane `HPC/SYnergy`, che continua a contenere:

- la facade Python `SYnergyQueue`;
- il bridge Cython `_synergy_submit.pyx`;
- il modulo nativo `_synergy_native`;
- la gestione del profiling;
- i test e gli esempi utente.

La fork di `dpctl` viene usata come dipendenza modificata, con lo scopo di aggiungere funzioni native posteriori richiamabili da `HPC/SYnergy`.

## Principio progettuale

La strategia scelta è non invasiva: non si modifica il comportamento originale della `submit` standard di `dpctl`.

Il percorso originale rimane invariato:

```text
dpctl.SyclQueue.submit(...)
    -> DPCTLQueue_SubmitRange / DPCTLQueue_SubmitNDRange
    -> sycl::queue::submit(...)
