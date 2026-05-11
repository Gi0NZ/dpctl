//===- dpctl_sycl_synergy_queue_interface.cpp -----------------------------===//
//
// Experimental SYnergy backend submit interface.
//
// This file adds SYnergy-specific submit entry points to the dpctl fork
// without modifying the standard DPCTLQueue_SubmitRange / SubmitNDRange path.
//
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_synergy_queue_interface.h"

#include <iostream>

extern "C" {

DPCTLSyclEventRef DPCTLQueue_SubmitRangeSYnergy(
    std::uintptr_t AdapterHandle,
    const DPCTLSyclKernelRef KRef,
    void **Args,
    const DPCTLKernelArgType *ArgTypes,
    std::size_t NArgs,
    const std::size_t Range[3],
    std::size_t NDims,
    const DPCTLSyclEventRef *DepEvents,
    std::size_t NDepEvents,
    unsigned int UncoreFrequency,
    unsigned int CoreFrequency,
    int UseFrequencyScaling)
{
    (void)AdapterHandle;
    (void)KRef;
    (void)Args;
    (void)ArgTypes;
    (void)NArgs;
    (void)Range;
    (void)NDims;
    (void)DepEvents;
    (void)NDepEvents;
    (void)UncoreFrequency;
    (void)CoreFrequency;
    (void)UseFrequencyScaling;

    std::cerr << "DPCTLQueue_SubmitRangeSYnergy placeholder called."
              << std::endl;

    return nullptr;
}

DPCTLSyclEventRef DPCTLQueue_SubmitNDRangeSYnergy(
    std::uintptr_t AdapterHandle,
    const DPCTLSyclKernelRef KRef,
    void **Args,
    const DPCTLKernelArgType *ArgTypes,
    std::size_t NArgs,
    const std::size_t GlobalRange[3],
    const std::size_t LocalRange[3],
    std::size_t NDims,
    const DPCTLSyclEventRef *DepEvents,
    std::size_t NDepEvents,
    unsigned int UncoreFrequency,
    unsigned int CoreFrequency,
    int UseFrequencyScaling)
{
    (void)AdapterHandle;
    (void)KRef;
    (void)Args;
    (void)ArgTypes;
    (void)NArgs;
    (void)GlobalRange;
    (void)LocalRange;
    (void)NDims;
    (void)DepEvents;
    (void)NDepEvents;
    (void)UncoreFrequency;
    (void)CoreFrequency;
    (void)UseFrequencyScaling;

    std::cerr << "DPCTLQueue_SubmitNDRangeSYnergy placeholder called."
              << std::endl;

    return nullptr;
}

}
