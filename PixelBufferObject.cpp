#include <gl/glew.h>
#include <cuda_gl_interop.h>

#include "PixelBufferObject.h"
#include "Logger.h"

PixelBufferObject::PixelBufferObject()
{
  glGenBuffers( 1, &mPboId );
}

PixelBufferObject::~PixelBufferObject()
{
  if ( mBound )
  {
    logger::Logger::Instance() << "PBO bound at destruction, id=" << mPboId;
  }

  if ( mCudaResource )
  {
    cudaError_t err = cudaGraphicsUnregisterResource( mCudaResource );
    if ( err != cudaSuccess )
    {
      logger::Logger::Instance() << "cudaGraphicsUnregisterResource failed during destruction of PixelBufferObject" << mPboId;
    }
  }

  glDeleteBuffers( 1, &mPboId );
}

void PixelBufferObject::Allocate( uint32_t byteCount )
{
  glBufferData( GL_PIXEL_UNPACK_BUFFER, byteCount, NULL, GL_DYNAMIC_COPY ); // last param always can be this one ?
}

void PixelBufferObject::BindPbo()
{
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, mPboId );
  mBound = true;
}

void PixelBufferObject::UnbindPbo()
{
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
  mBound = false;
}

uint8_t* PixelBufferObject::MapPboBuffer()
{
  return reinterpret_cast<uint8_t*>( glMapBuffer( GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY ) );
}

void PixelBufferObject::UnmapPboBuffer()
{
  glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );
}

void PixelBufferObject::RegisterCudaResource()
{
  cudaError_t err = cudaGraphicsGLRegisterBuffer( &mCudaResource, mPboId, cudaGraphicsMapFlagsNone );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( "cudaGraphicsGLRegisterBuffer failed" );
  }
}

void PixelBufferObject::MapCudaResource()
{
  cudaError_t err = cudaGraphicsMapResources( 1, &mCudaResource );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( "cudaGraphicsMapResources failed" );
  }
}

void PixelBufferObject::UnmapCudaResource()
{
  cudaError_t err = cudaGraphicsUnmapResources( 1, &mCudaResource );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( "cudaGraphicsUnmapResources failed" );
  }
}

uint8_t* PixelBufferObject::GetCudaMappedPointer()
{
  uint8_t* ptr = nullptr;
  size_t mapped_size = 0;
  cudaError_t err = cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void**>( &ptr ), &mapped_size, mCudaResource );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( "cudaGraphicsResourceGetMappedPointer failed" );
  }
  return ptr;
}
