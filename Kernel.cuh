#pragma once

#include <cstdint>
#include <cuda_runtime.h>

cudaError_t RunFillKernel( uint32_t* buffer, const uint32_t value, const uint32_t width, const uint32_t height );
cudaError_t RunStepKernel( uint32_t* frontBuffer, uint32_t* backBuffer, uint32_t width, uint32_t height, const uint32_t livingColor, const uint32_t deadColor );
cudaError_t RunRandomKernel( uint32_t* buffer, const float prob, const uint32_t width, const uint32_t height, const uint32_t livingColor, const uint32_t deadColor);
