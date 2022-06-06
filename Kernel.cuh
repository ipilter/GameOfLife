#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t RunFillKernel( unsigned char* buffer, const unsigned char value, const int width, const int height );
cudaError_t RunStepKernel( unsigned char* pos, int width, int height );