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
    logger::Logger::instance() << "PBO bound at destruction, id=" << mPboId;
  }

  if ( mCudaResource )
  {
    cudaError_t err = cudaGraphicsUnregisterResource( mCudaResource );
    if ( err != cudaSuccess )
    {
      logger::Logger::instance() << "cudaGraphicsUnregisterResource failed during destruction of PixelBufferObject" << mPboId;
    }
  }

  glDeleteBuffers( 1, &mPboId );
}

void PixelBufferObject::allocate( uint32_t byteCount )
{
  glBufferData( GL_PIXEL_UNPACK_BUFFER, byteCount, NULL, GL_DYNAMIC_COPY ); // last param always can be this one ?
}

void PixelBufferObject::bindPbo()
{
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, mPboId );
  mBound = true;
}

void PixelBufferObject::unbindPbo()
{
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
  mBound = false;
}

uint8_t* PixelBufferObject::mapPboBuffer()
{
  return reinterpret_cast<uint8_t*>( glMapBuffer( GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY ) );
}

void PixelBufferObject::unmapPboBuffer()
{
  glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );
}

void PixelBufferObject::registerCudaResource()
{
  cudaError_t err = cudaGraphicsGLRegisterBuffer( &mCudaResource, mPboId, cudaGraphicsMapFlagsNone );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( "cudaGraphicsGLRegisterBuffer failed" );
  }
}

void PixelBufferObject::mapCudaResource()
{
  cudaError_t err = cudaGraphicsMapResources( 1, &mCudaResource );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( "cudaGraphicsMapResources failed" );
  }
}

void PixelBufferObject::unmapCudaResource()
{
  cudaError_t err = cudaGraphicsUnmapResources( 1, &mCudaResource );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( "cudaGraphicsUnmapResources failed" );
  }
}

uint8_t* PixelBufferObject::getCudaMappedPointer()
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
