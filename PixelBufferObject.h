#pragma once

#include <cstdint>
#include <memory>
#include <driver_types.h>

// pixel buffer object with cuda resource
class PixelBufferObject
{
public:
  using Ptr = std::unique_ptr<PixelBufferObject>;

public:
  PixelBufferObject();
  ~PixelBufferObject();

  void allocate( uint32_t byteCount );

  void bindPbo();
  void unbindPbo();

  uint8_t* mapPboBuffer();
  void unmapPboBuffer();

  // cuda
  void registerCudaResource();
  void mapCudaResource();
  void unmapCudaResource();
  uint8_t* getCudaMappedPointer();

private:
  uint32_t mPboId = 0;
  cudaGraphicsResource_t mCudaResource = 0;
  bool mBound = false;
};