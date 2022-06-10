#pragma once

#include <cstdint>
#include <cuda_runtime.h>

cudaError_t RunFillKernel( uint8_t* buffer, const uint8_t value, const uint32_t width, const uint32_t height );
cudaError_t RunStepKernel( uint8_t* frontBuffer, uint8_t* backBuffer, uint32_t width, uint32_t height );