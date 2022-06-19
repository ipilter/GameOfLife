
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Kernel.cuh"

__device__ const uint32_t& Max( const uint32_t& a, const uint32_t& b )
{
  return a >= b ? a : b;
}

__device__ uint8_t Component( const uint32_t& color, const uint32_t& idx )
{
  switch ( idx )
  {
    case 0:
      return static_cast<uint8_t>( ( color & 0x000000FF ) >> 0 );
    case 1:
      return static_cast<uint8_t>( ( color & 0x0000FF00 ) >> 8 );
    case 2:
      return static_cast<uint8_t>( ( color & 0x00FF0000 ) >> 16 );
    case 3:
      return static_cast<uint8_t>( ( color & 0xFF000000 ) >> 24 );
    default:
      return 0;
  }
}

__device__ uint32_t Color( const uint8_t r = 0, const uint8_t g = 0, const uint8_t b = 0, const uint8_t a = 255 )
{
  return ( r << 0 ) | ( g << 8 ) | ( b << 16 ) | ( a << 24 );
}

__device__ uint32_t GetValue( uint32_t* buffer, int32_t x, int32_t y, uint32_t width, uint32_t height )
{
  const int32_t rx = x < 0 ? width - 1 : x >= width ? 0 : x;
  const int32_t ry = y < 0 ? height - 1 : y >= height ? 0 : y;
  return buffer[rx + ry * width]; // RGBA
}

__global__ void StepKernel( uint32_t* frontBuffer, uint32_t* backBuffer, const uint32_t width, const uint32_t height, const uint32_t livingColor, const uint32_t deadColor )
{
  static const bool mDecideData[] = {
  //0  1  2  3  4  5  6  7  8    living neighbour count
    0, 0, 0, 1, 0, 0, 0, 0, 0,   // dead cell new state |  8
    0, 0, 1, 1, 0, 0, 0, 0, 0 }; // live cell new state | 12

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
  {
    return;
  }

  const uint32_t current = ( GetValue( frontBuffer, x, y, width, height ) == deadColor ? 0 : 1 );
  const uint32_t sum = ( GetValue( frontBuffer, x - 1, y - 1, width, height ) == deadColor ? 0 : 1 ) +
                       ( GetValue( frontBuffer, x    , y - 1, width, height ) == deadColor ? 0 : 1 ) +
                       ( GetValue( frontBuffer, x + 1, y - 1, width, height ) == deadColor ? 0 : 1 ) +
                       ( GetValue( frontBuffer, x - 1, y    , width, height ) == deadColor ? 0 : 1 ) +
                       ( GetValue( frontBuffer, x + 1, y    , width, height ) == deadColor ? 0 : 1 ) +
                       ( GetValue( frontBuffer, x - 1, y + 1, width, height ) == deadColor ? 0 : 1 ) +
                       ( GetValue( frontBuffer, x    , y + 1, width, height ) == deadColor ? 0 : 1 ) +
                       ( GetValue( frontBuffer, x + 1, y + 1, width, height ) == deadColor ? 0 : 1 );
  backBuffer[x + width * y] = ( mDecideData[current * 9 + sum] ? livingColor : deadColor );
}

__global__ void FillKernel( uint32_t* buffer, const uint32_t width, const uint32_t height, const uint32_t value )
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x >= width || y >= height )
  {
    return;
  }
  const size_t offset = x + width * y;
  buffer[offset] = value;
}

__global__ void InitRandom( unsigned int seed, const uint32_t width, const uint32_t height, curandState_t* states )
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x >= width || y >= height )
  {
    return;
  }

  /* we have to initialize the state */
  curand_init( seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
               x, /* the sequence number should be different for each core (unless you want all
                           cores to get the same sequence of numbers for some reason - use thread id! */
               y, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
               &states[x + y * width] );
}

__global__ void RandomKernel( uint32_t* buffer, const uint32_t width, const uint32_t height, const uint32_t livingColor, const uint32_t deadColor, const float prob, curandState_t* states )
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x >= width || y >= height )
  {
    return;
  }

  const float r = curand_uniform( &states[x + y * width] );
  uint32_t v = deadColor;
  if ( r > prob )
  {
    v = livingColor;
  }

  buffer[x + width * y + 0] = Max(v, buffer[x + width * y + 0]);
}

cudaError_t RunFillKernel( uint32_t* buffer, const uint32_t value, const uint32_t width, const uint32_t height )
{
  const dim3 blockSize( 32, 32 );// number of threads per block along x/y-axis
  const dim3 gridSize( ( width + blockSize.x - 1 ) / blockSize.x
                       , ( height + blockSize.y - 1 ) / blockSize.y ); // number of blocks in the grid

  //cudaEvent_t start, stop;
  //cudaEventCreate( &start );
  //cudaEventCreate( &stop );
  //
  //cudaEventRecord( start, 0 );
  FillKernel<<<gridSize, blockSize>>>( buffer, width, height, value );
  //cudaEventRecord( stop, 0 );
  //cudaEventSynchronize( stop );
  //
  //float time = 0.0f;
  //cudaEventElapsedTime( &time, start, stop );

  return cudaGetLastError();
}

cudaError_t RunStepKernel( uint32_t* frontBuffer, uint32_t* backBuffer, uint32_t width, uint32_t height, const uint32_t livingColor, const uint32_t deadColor )
{
  const dim3 blockSize( 32, 32 );// number of threads per block along x/y-axis
  const dim3 gridSize( (width + blockSize.x - 1) / blockSize.x
                       , (height + blockSize.y - 1) / blockSize.y ); // number of blocks in the grid

  //cudaEvent_t start, stop;
  //cudaEventCreate( &start );
  //cudaEventCreate( &stop );
  //
  //cudaEventRecord( start, 0 );
  StepKernel<<<gridSize, blockSize>>>( frontBuffer, backBuffer, width, height, livingColor, deadColor );
  //cudaEventRecord( stop, 0 );
  //cudaEventSynchronize( stop );
  //
  //float time = 0.0f;
  //cudaEventElapsedTime( &time, start, stop );
  return cudaGetLastError();
}

cudaError_t RunRandomKernel( uint32_t* buffer, const float prob, const uint32_t width, const uint32_t height, const uint32_t livingColor, const uint32_t deadColor )
{
  const dim3 blockSize( 32, 32 );// number of threads per block along x/y-axis
  const dim3 gridSize( (width + blockSize.x - 1) / blockSize.x
                       , (height + blockSize.y - 1) / blockSize.y ); // number of blocks in the grid

  // TODO: revisit this random number stuff. Random states per pixel?
  curandState_t* states = nullptr;
  cudaMalloc((void**) &states, width * height * sizeof(curandState_t));

  InitRandom<<<gridSize, blockSize>>>( static_cast<uint32_t>( time( nullptr ) ), width, height, states );
  RandomKernel<<<gridSize, blockSize>>>( buffer, width, height, livingColor, deadColor, prob, states );

  cudaFree( states );

  return cudaGetLastError();
}
