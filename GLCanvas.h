#pragma once

#include <map>
#include <vector>
#include <gl/glew.h>
#include "WxMain.h"
#include <wx/glcanvas.h>
#include "Math.h"
#include "cuda_gl_interop.h"
#include "cuda_runtime.h"
#include "Logger.h"

// pattern storage
struct Pattern
{
  Pattern( GLuint w, GLuint h, const std::vector<bool>& bits )
    : mWidth( w )
    , mHeight( h )
    , mBits( bits )
  {}

  bool at( const GLuint x, const GLuint y ) const
  {
    return mBits[x + mWidth * y];
  }

  GLuint width() const
  {
    return mWidth;
  }

  GLuint height() const
  {
    return mHeight;
  }

private:
  GLuint mWidth;
  GLuint mHeight;
  std::vector<bool> mBits;
};

struct Texture
{
  Texture( int w, int h )
    : mWidth( w )
    , mHeight( h )
  {
    glGenTextures( 1, &mId );
    glBindTexture( GL_TEXTURE_2D, mId );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
    glBindTexture( GL_TEXTURE_2D, 0 );
  }

  ~Texture()
  {
    glDeleteTextures( 1, &mId );
  }

  void bind()
  {
    glBindTexture( GL_TEXTURE_2D, mId );
  }

  void unbind()
  {
    glBindTexture( GL_TEXTURE_2D, 0 );
  }

  void bindTextureUnit( GLuint unitId )
  {
    glBindTextureUnit( 0, mId );
  }

  void unbindTextureUnit()
  {
    glBindTextureUnit( 0, 0 );
  }

  // allocate and copy copy pixels from bound PBO
  void createFromPBO()
  {
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
  }

  void updateFromPBO()
  {
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
  }

  void updateFromPBO(GLuint regionPosX, GLuint regionPosY, GLuint regionWidth, GLuint regionHeight)
  {
    glPixelStorei( GL_UNPACK_ROW_LENGTH, mWidth );
    glTexSubImage2D( GL_TEXTURE_2D, 0, regionPosX, regionPosY, regionWidth, regionHeight, GL_RGBA, GL_UNSIGNED_BYTE
                     , reinterpret_cast<void*>( static_cast<size_t>( regionPosX ) * 4 + regionPosY * 4 * mWidth ) );
    glPixelStorei( GL_UNPACK_ROW_LENGTH, 0 );
  }

  int width() const
  {
    return mWidth;
  }

  int height() const
  {
    return mHeight;
  }

private:
  GLuint mId = 0;
  int mWidth = 0;
  int mHeight = 0;
};

// pixel buffer object with cuda resource
struct PBO
{
  // pbo
  PBO()
  {
    glGenBuffers( 1, &mPboId );
  }

  ~PBO()
  {
    if ( mBound )
    {
      logger::Logger::instance() << "PBO bound at destruction, id=" << mPboId;  
    }

    glDeleteBuffers( 1, &mPboId );
    if ( mCudaResource )
    {
      cudaGraphicsUnregisterResource( mCudaResource );
    }
  }

  void allocate( GLuint byteCount )
  {
    glBufferData( GL_PIXEL_UNPACK_BUFFER, byteCount, NULL, GL_STATIC_DRAW ); //GL_STATIC_DRAW ??
  }

  void bindPbo()
  {
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, mPboId );
    glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
    mBound = true;
  }

  void unbindPbo()
  {
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    mBound = false;
  }

  GLubyte* mapPboBuffer()
  {
    return reinterpret_cast<GLubyte*>( glMapBuffer( GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY ) );
  }

  void unmapPboBuffer()
  {
    glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );
  }

  // cuda
  void registerCudaResource()
  {
    cudaError_t err = cudaGraphicsGLRegisterBuffer( &mCudaResource, mPboId, cudaGraphicsMapFlagsNone );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( "cudaGraphicsMapResources failed" );
    }
  }

  void mapCudaResource()
  {

    cudaError_t err = cudaGraphicsMapResources( 1, &mCudaResource );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( "cudaGraphicsMapResources failed" );
    }
  }

  void unmapCudaResource()
  {
    cudaError_t err = cudaGraphicsUnmapResources( 1, &mCudaResource );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( "cudaGraphicsMapResources failed" );
    }
  }

  std::tuple<unsigned char*, size_t> getCudaMappedPointer()
  {
    unsigned char* ptr = nullptr;
    size_t mapped_size;
    cudaError_t err = cudaGraphicsResourceGetMappedPointer( (void**) &ptr, &mapped_size, mCudaResource );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( "cudaGraphicsMapResources failed" );
    }
    return std::make_tuple( ptr, mapped_size );
  }

  void bindCudaResource()
  {
    cudaError_t err = cudaSuccess;
    err = cudaGraphicsMapResources( 1, &mCudaResource );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( "cudaGraphicsMapResources failed" );
    }
  }

  void unbindCudaResource()
  {
    cudaError_t err = cudaSuccess;
    err = cudaGraphicsMapResources( 1, &mCudaResource );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( "cudaGraphicsMapResources failed" );
    }
  }

private:
  GLuint mPboId = 0;
  cudaGraphicsResource_t mCudaResource = 0;
  bool mBound = false;
};

class GLCanvas : public wxGLCanvas
{
public:
  GLCanvas( const int textureExponent,
            wxWindow* parent,
            wxWindowID id = wxID_ANY,
            const int* attribList = 0,
            const wxPoint& pos = wxDefaultPosition,
            const wxSize& size = wxDefaultSize,
            long style = 0L,
            const wxString& name = L"GLCanvas",
            const wxPalette& palette = wxNullPalette );

  virtual ~GLCanvas();

  GLCanvas( const GLCanvas& ) = delete;
  GLCanvas( GLCanvas&& ) = delete;
  GLCanvas& operator = ( const GLCanvas& ) = delete;
  GLCanvas& operator = ( GLCanvas&& ) = delete;

  void SetDrawColor( const math::uvec3& color );
  const math::uvec3& GetDrawColor() const;

  void Step();
  void Reset();

private:
  void InitializeGLEW();
  void CreateGeometry();
  void CreateShaderProgram();
  void CreateTexture();
  GLuint CreateShader( GLuint kind, const std::string& src );

  math::vec2 ScreenToWorld( const math::vec2& screenSpacePoint );
  math::ivec2 WorldToImage( const math::vec2& worldSpacePoint );

  void SetPixel( const math::ivec2& pixel );
  void OnPaint( wxPaintEvent& event );
  void OnSize( wxSizeEvent& event );
  void OnMouseMove( wxMouseEvent& event );
  void OnMouseRightDown( wxMouseEvent& event );
  void OnMouseRightUp( wxMouseEvent& event );
  void OnMouseLeftDown( wxMouseEvent& event );
  void OnMouseLeftUp( wxMouseEvent& event );
  void OnMouseLeave( wxMouseEvent& event );
  void OnMouseWheel( wxMouseEvent& event );

  // opengl context
  std::unique_ptr<wxGLContext> mContext;

  // parameters
  float mQuadSize;
  int mTextureExponent;

  // mesh
  GLuint mVbo;
  GLuint mIbo;
  GLuint mVao;

  std::vector<std::unique_ptr<Texture>> mTextures;

  // pixel buffer list
  std::vector<std::unique_ptr<PBO>> mPBOs;

  // shader
  GLuint mVertexShader = 0;
  GLuint mFragmentxShader = 0;
  GLuint mShaderProgram = 0;

  // view
  math::mat4 mProjectionMatrix;
  math::mat4 mViewMatrix;

  // control
  bool mPanningActive;
  math::vec2 mPreviousMousePosition;
  bool mDrawingActive;
  math::uvec3 mDrawColor;

  // pattern selection
  std::vector<std::unique_ptr<Pattern>> mDrawPatterns;
};
