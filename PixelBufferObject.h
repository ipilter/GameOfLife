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

  void Allocate( uint32_t byteCount );

  void BindPbo();
  void UnbindPbo();

  uint32_t* MapPboBuffer();
  void UnmapPboBuffer();

  // cuda
  void RegisterCudaResource();
  void MapCudaResource();
  void UnmapCudaResource();
  uint32_t* GetCudaMappedPointer();

private:
  uint32_t mPboId = 0;
  cudaGraphicsResource_t mCudaResource = 0;
  bool mBound = false;
};
