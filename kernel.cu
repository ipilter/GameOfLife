
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <cmath>

__device__ unsigned char getValue( unsigned char* rgb, int x, int y, int width, int height )
{
  int rx = x;
  int ry = y;

  if ( x < 0 )
  {
    rx = width - 1;
  }
  else if ( x >= width )
  {
    rx = 0;
  }

  if ( y < 0 )
  {
    ry = height - 1;
  }
  else if ( y >= height )
  {
    ry = 0;
  }

  return rgb[rx * 4 + ry * 4 * width + 0]; // first component from BGRA
}

__global__ void StepKernel( unsigned char* rgb, int const width, int const height )
{
  const int mDecideData[] = {
  //0  1  2  3  4  5  6  7  8    living neighbour count
    0, 0, 0, 1, 0, 0, 0, 0, 0,   // dead cell new state
    0, 0, 1, 1, 0, 0, 0, 0, 0 }; // live cell new state

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
  {
    return;
  }

  unsigned char current = (getValue( rgb, x, y, width, height ) == 255 ? 1 : 0);
  int sum = (getValue( rgb, x-1, y-1, width, height ) == 255 ? 1 : 0) +
            (getValue( rgb,   x, y-1, width, height ) == 255 ? 1 : 0) +
            (getValue( rgb, x+1, y-1, width, height ) == 255 ? 1 : 0) +
            (getValue( rgb, x-1,   y, width, height ) == 255 ? 1 : 0) +
            (getValue( rgb, x+1,   y, width, height ) == 255 ? 1 : 0) +
            (getValue( rgb, x-1, y+1, width, height ) == 255 ? 1 : 0) +
            (getValue( rgb,   x, y+1, width, height ) == 255 ? 1 : 0) +
            (getValue( rgb, x+1, y+1, width, height ) == 255 ? 1 : 0);

  const unsigned char newState = (mDecideData[current * 9 + sum] > 0 ? 255 : 0);
  const int offset = x * 4 + width * y * 4;
  rgb[offset + 0] = newState;
  rgb[offset + 1] = newState;
  rgb[offset + 2] = newState;

  //const int offset = x * 3 + width * y * 3;
  //rgb[offset + 0] = 255 * (x / float(width));
  //rgb[offset + 1] = 55;
  //rgb[offset + 2] = 255 * (y / float(height));
}

__global__ void FillKernel( unsigned char* buffer, const int width, const int height, const unsigned char value )
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x >= width || y >= height )
  {
    return;
  }
  const int offset = x * 4 + width * y * 4;
  buffer[offset + 0] = value;
  buffer[offset + 1] = value;
  buffer[offset + 2] = value;
  buffer[offset + 3] = value;
}

cudaError_t RunFillKernel(unsigned char* buffer, const unsigned char value, const int width, const int height)
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

cudaError_t RunStepKernel( unsigned char* rgb, int width, int height )
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
  StepKernel <<<blocksPerGrid, threadsPerBlock>>> ( rgb, width, height );
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );
  return cudaGetLastError();
}
