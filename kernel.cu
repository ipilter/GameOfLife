
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include <device_launch_parameters.h>

#include "Kernel.cuh"

__device__ uint8_t getValue( uint8_t* buffer, int32_t x, int32_t y, uint32_t width, uint32_t height )
{
  const int32_t rx = x < 0 ? width - 1 : x >= width ? 0 : x;
  const int32_t ry = y < 0 ? height - 1 : y >= height ? 0 : y;
  return buffer[rx * 4 + ry * 4 * width]; // first component from BGRA
}

__global__ void StepKernel( uint8_t* frontBuffer, uint8_t* backBuffer, const uint32_t width, const uint32_t height )
{
  const bool mDecideData[] = {
  //0  1  2  3  4  5  6  7  8    living neighbour count
    0, 0, 0, 1, 0, 0, 0, 0, 0,   // dead cell new state
    0, 0, 1, 1, 0, 0, 0, 0, 0 }; // live cell new state

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
  {
    return;
  }

  const  uint8_t current = (getValue( frontBuffer, x, y, width, height ) == 255 ? 1 : 0);
  const  uint32_t sum = (getValue( frontBuffer, x-1, y-1, width, height ) == 255 ? 1 : 0) +
                        (getValue( frontBuffer,   x, y-1, width, height ) == 255 ? 1 : 0) +
                        (getValue( frontBuffer, x+1, y-1, width, height ) == 255 ? 1 : 0) +
                        (getValue( frontBuffer, x-1,   y, width, height ) == 255 ? 1 : 0) +
                        (getValue( frontBuffer, x+1,   y, width, height ) == 255 ? 1 : 0) +
                        (getValue( frontBuffer, x-1, y+1, width, height ) == 255 ? 1 : 0) +
                        (getValue( frontBuffer,   x, y+1, width, height ) == 255 ? 1 : 0) +
                        (getValue( frontBuffer, x+1, y+1, width, height ) == 255 ? 1 : 0);

  const uint8_t newState = (mDecideData[current * 9 + sum] > 0 ? 255 : 0);
  const uint32_t offset = x * 4 + width * y * 4;
  backBuffer[offset + 0] = newState;
  backBuffer[offset + 1] = newState;
  backBuffer[offset + 2] = newState;

  // rainbow test pixels
  //backBuffer[offset + 0] = 255 * (x / float(width));
  //backBuffer[offset + 1] = 55;
  //backBuffer[offset + 2] = 255 * (y / float(height));
}

__global__ void FillKernel( uint8_t* buffer, const uint32_t width, const uint32_t height, const uint8_t value )
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x >= width || y >= height )
  {
    return;
  }
  const size_t offset = x * 4ull + width * y * 4ull;
  buffer[offset + 0] = value;
  buffer[offset + 1] = value;
  buffer[offset + 2] = value;
  buffer[offset + 3] = value;
}

cudaError_t RunFillKernel(uint8_t* buffer, const uint8_t value, const uint32_t width, const uint32_t height)
{
  dim3 threadsPerBlock( 32, 32, 1 );
  dim3 blocksPerGrid( width / threadsPerBlock.x, height / threadsPerBlock.y, 1 ); // TODO works only with power of 2 texture sizes !!
  if ( blocksPerGrid.x == 0 )
  {
    blocksPerGrid.x = 1;
  }
  if ( blocksPerGrid.y == 0 )
  {
    blocksPerGrid.y = 1;
  }

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );
  FillKernel <<<blocksPerGrid, threadsPerBlock>>> ( buffer, width, height, value );
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );

  return cudaGetLastError();
}

cudaError_t RunStepKernel( uint8_t* frontBuffer, uint8_t* backBuffer, uint32_t width, uint32_t height )
{
  dim3 threadsPerBlock( 32, 32, 1 );
  dim3 blocksPerGrid( width / threadsPerBlock.x, height / threadsPerBlock.y, 1 ); // TODO works only with power of 2 texture sizes !!
  if ( blocksPerGrid.x == 0 )
  {
    blocksPerGrid.x = 1;
  }
  if ( blocksPerGrid.y == 0 )
  {
    blocksPerGrid.y = 1;
  }

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );
  StepKernel <<<blocksPerGrid, threadsPerBlock>>> ( frontBuffer, backBuffer, width, height );
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );
  return cudaGetLastError();
}
