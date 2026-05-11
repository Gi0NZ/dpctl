#pragma once

#include "dpctl_sycl_queue_interface.h"

#include <cstddef>
#include <cstdint>

extern "C" {

DPCTL_API
__dpctl_give DPCTLSyclEventRef DPCTLQueue_SubmitRangeSYnergy(
    std::uintptr_t AdapterHandle,
    __dpctl_keep const DPCTLSyclKernelRef KRef,
    void **Args,
    const DPCTLKernelArgType *ArgTypes,
    std::size_t NArgs,
    const std::size_t Range[3],
    std::size_t NDims,
    __dpctl_keep const DPCTLSyclEventRef *DepEvents,
    std::size_t NDepEvents,
    unsigned int UncoreFrequency,
    unsigned int CoreFrequency,
    int UseFrequencyScaling);

DPCTL_API
__dpctl_give DPCTLSyclEventRef DPCTLQueue_SubmitNDRangeSYnergy(
    std::uintptr_t AdapterHandle,
    __dpctl_keep const DPCTLSyclKernelRef KRef,
    void **Args,
    const DPCTLKernelArgType *ArgTypes,
    std::size_t NArgs,
    const std::size_t GlobalRange[3],
    const std::size_t LocalRange[3],
    std::size_t NDims,
    __dpctl_keep const DPCTLSyclEventRef *DepEvents,
    std::size_t NDepEvents,
    unsigned int UncoreFrequency,
    unsigned int CoreFrequency,
    int UseFrequencyScaling);

}
