#include <vector>
#include <sstream>
#include <algorithm>
#include <filesystem>

#include "timer.h"
#include "MainFrame.h"
#include "GLCanvas.h"

#include "Util.h"
#include "Logger.h"
#include "kernel.cuh"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

GLCanvas::GLCanvas( const uint32_t textureSize
                    , wxWindow* parent
                    , wxWindowID id
                    , const int32_t* attribList
                    , const wxPoint& pos
                    , const wxSize& size
                    , long style
                    , const wxString& name
                    , const wxPalette& palette )
  : wxGLCanvas( parent, id, attribList, pos, size, style, name, palette )
  , mQuadSize( 1.0f )
  , mTextureSize( textureSize )
  , mPrimaryColor ( util::Color( 255, 255, 255 ) )
  , mSecondaryColor ( util::Color( 0, 0, 0 ) )
  , mCurrentDrawingColor ( util::Color( 255, 255, 255 ) ) // TODO index in an array instead
{
  InitializeEventHandlers();
  InitializeGL();
  InitializeCuda();
  CreatePatterns();
  CreatePixelBuffers();
  CreateTextures();
  CreateGeometry();
  CreateShaders();

  mViewMatrix = glm::translate( glm::scale( glm::identity<math::mat4>(), math::vec3( 5.0f ) ), math::vec3( -mQuadSize / 2.0f, -mQuadSize / 2.0f, 0.0f ) ); // at center zoom level 5
  mStepTimer = std::make_unique<StepTimer>( this );
}

GLCanvas::~GLCanvas()
{
  //{
  //  std::ofstream patternsStream( "e:\\patterns.dat" );
  //  util::Write_t( patternsStream, mDrawPatterns.size() );
  //  for ( const auto& p : mDrawPatterns )
  //  {
  //    p->Write( patternsStream );
  //  }
  //}

  SetCurrent( *mContext );

  glDeleteShader( mVertexShader );
  glDeleteShader( mFragmentxShader );
  glDeleteProgram( mShaderProgram );
  
  FreeCudaRandomStates();
}

void GLCanvas::SetPrimaryColor( const uint32_t color )
{
  mPrimaryColor = color;
}

const uint32_t GLCanvas::GetPrimaryColor() const
{
  return mPrimaryColor;
}

void GLCanvas::SetSecondaryColor( const uint32_t color )
{
  mSecondaryColor = color;
}

const uint32_t GLCanvas::GetSecondaryColor() const
{
  return mSecondaryColor;
}

void GLCanvas::SetCurrentPattern( const uint32_t idx )
{
  mDrawPatternIdx = idx;
  
  Refresh();
}

uint32_t GLCanvas::GetPatternCount() const
{
  return mDrawPatterns.size();
}

const Pattern& GLCanvas::GetPattern( const uint32_t idx ) const
{
  return *mDrawPatterns[idx]->mPattern;
}

void GLCanvas::SetDrawPixelGrid( const bool drawPixelGrid )
{
  mDrawPixelGrid = drawPixelGrid;

  Refresh();
}

void GLCanvas::InitializeEventHandlers()
{
  Bind( wxEVT_SIZE, &GLCanvas::OnSize, this );
  Bind( wxEVT_PAINT, &GLCanvas::OnPaint, this );
  Bind( wxEVT_RIGHT_DOWN, &GLCanvas::OnMouseRightDown, this );
  Bind( wxEVT_RIGHT_UP, &GLCanvas::OnMouseRightUp, this );
  Bind( wxEVT_MIDDLE_DOWN, &GLCanvas::OnMouseMiddleDown, this );
  Bind( wxEVT_MIDDLE_UP, &GLCanvas::OnMouseMiddleUp, this );
  Bind( wxEVT_LEFT_DOWN, &GLCanvas::OnMouseLeftDown, this );
  Bind( wxEVT_LEFT_UP, &GLCanvas::OnMouseLeftUp, this );
  Bind( wxEVT_MOTION, &GLCanvas::OnMouseMove, this );
  Bind( wxEVT_LEAVE_WINDOW, &GLCanvas::OnMouseLeave, this );
  Bind( wxEVT_MOUSEWHEEL, &GLCanvas::OnMouseWheel, this );
  Bind( wxEVT_KEY_DOWN, &GLCanvas::OnKeyDown, this );
}

void GLCanvas::InitializeGL()
{
  wxGLContextAttrs contextAttrs;
  contextAttrs.CoreProfile().OGLVersion( 4, 5 ).Robust().ResetIsolation().EndList();
  mContext = std::make_unique<wxGLContext>( this, nullptr, &contextAttrs );
  if ( !SetCurrent( *mContext ) )
  {
    throw std::exception( "Cannot set GL context" );
  }

  glewExperimental = false;
  GLenum err = glewInit();
  if ( err != GLEW_OK )
  {
    const uint8_t* msg = glewGetErrorString( err );
    throw std::exception( reinterpret_cast<const char*>( msg ) );
  }

  const bool fp64( glewGetExtension( "GL_ARB_gpu_shader_fp64" ) );
  logger::Logger::Instance() << "GL_ARB_gpu_shader_fp64 " << ( fp64 == 1 ? "supported" : "not supported" ) << "\n";

  int32_t maxTextureSize = 0;
  glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTextureSize );
  logger::Logger::Instance() << "Max texture size on current GPU " << maxTextureSize << "x" << maxTextureSize << "\n";
  mTextureSize = glm::min( mTextureSize, static_cast<uint32_t>( maxTextureSize ) ); // nxn pixels
  mTexturePatternSize = 10; // nxn pixels
}

void GLCanvas::InitializeCuda()
{
  cudaError_t error_test = cudaSuccess;

  int32_t gpuCount = 0;
  error_test = cudaGetDeviceCount( &gpuCount );
  if ( error_test != cudaSuccess )
  {
    throw std::runtime_error( "cudaGetDeviceCount failed" );
  }

  cudaDeviceProp prop = { 0 };
  int32_t gpuId = 0;
  error_test = cudaGetDeviceProperties( &prop, gpuId );
  if ( error_test != cudaSuccess )
  {
    throw std::runtime_error( "cudaGetDeviceProperties failed" );
  }

  error_test = cudaGLSetGLDevice( gpuId );
  if ( error_test != cudaSuccess )
  {
    throw std::runtime_error( "cudaGLSetGLDevice failed" );
  }
}

void GLCanvas::CreatePixelBuffers()
{
  int32_t pboAlignment = -1;
  glGetIntegerv( GL_UNPACK_ALIGNMENT, &pboAlignment );
  if ( pboAlignment != 4 )
  {
    std::stringstream ss;
    ss << "invalid PBO alignment: " << pboAlignment;
    throw std::runtime_error( ss.str() );
  }

  mPBOs.push_back( std::make_unique<PixelBufferObject>() );
  mFrontBufferIdx = mPBOs.size() - 1;

  mPBOs.push_back( std::make_unique<PixelBufferObject>() );
  mBackBufferIdx = mPBOs.size() - 1;
}

void GLCanvas::CreateGeometry()
{
  // world quad
  {
    const std::vector<float> points = { 0.0f,      0.0f,       0.0f, 1.0f // vtx bl
                                       , mQuadSize, 0.0f,      1.0f, 1.0f // vtx br
                                       , 0.0f,      mQuadSize, 0.0f, 0.0f // vtx tl
                                       , mQuadSize, mQuadSize, 1.0f, 0.0f // vtx tr
    };

    const std::vector<uint32_t> indices = { 0, 1, 2,  1, 3, 2 };  // triangle vertex indices

    // layout of data inside points array
    const size_t stride = 4 * sizeof( float );
    const size_t vertexOffset = 0;
    const size_t texelOffset = 2 * sizeof( float );

    mMeshes.push_back( std::make_unique<Mesh>( points, indices, stride, vertexOffset, texelOffset ) );
  }

  // pixel grid quad
  {
    const float pixelPatternTexelSize = mTextureSize / static_cast<float>( mTexturePatternSize );
    const std::vector<float> points = { 0.0f,      0.0f,       0.0,                   pixelPatternTexelSize // vtx bl
                                       , mQuadSize, 0.0f,      pixelPatternTexelSize, pixelPatternTexelSize // vtx br
                                       , 0.0f,      mQuadSize, 0.0,                   0.0                   // vtx tl
                                       , mQuadSize, mQuadSize, pixelPatternTexelSize, 0.0                   // vtx tr
    };

    const std::vector<uint32_t> indices = { 0, 1, 2,  1, 3, 2 };  // triangle vertex indices

    // layout of data inside points array
    const size_t stride = 4 * sizeof( float );
    const size_t vertexOffset = 0;
    const size_t texelOffset = 2 * sizeof( float );

    mMeshes.push_back( std::make_unique<Mesh>( points, indices, stride, vertexOffset, texelOffset ) );
  }

  // create pattern quads
  {
    for ( const auto& patternInfo : mDrawPatterns )
    {
      const float pixelSize = 1.0f / mTextureSize;
      const float quadWidth = pixelSize * patternInfo->mPattern->Width(); // TODO precision? use 1 x 1 * aspect quad and scale it in shader
      const float quadHeight = pixelSize * patternInfo->mPattern->Height();

      // invert y to match texture`s pixel y direction // TODO better approach than negative y coordinates, reason for negative is to match textre`s pixel y direction. See SetPixel()?
      const std::vector<float> points = { 0.0f,       0.0f,       0.0f, 0.0f // vtx tl
                                         , quadWidth, 0.0f,       1.0f, 0.0f // vtx tr
                                         , 0.0f,      -quadHeight, 0.0f, 1.0f // vtx bl
                                         , quadWidth, -quadHeight, 1.0f, 1.0f // vtx br
      };

      const std::vector<uint32_t> indices = { 0, 1, 2,  1, 3, 2 };  // triangle vertex indices

      // layout of data inside points array
      const size_t stride = 4 * sizeof( float );
      const size_t vertexOffset = 0;
      const size_t texelOffset = 2 * sizeof( float );
      mMeshes.push_back( std::make_unique<Mesh>( points, indices, stride, vertexOffset, texelOffset ) ); 
      patternInfo->mMeshIdx = mMeshes.size() - 1;
    }
  }
}

void GLCanvas::CreateShaders()
{
  mShaderProgram = glCreateProgram();
  if ( mShaderProgram == 0 )
  {
    throw std::runtime_error( "cannot create shader program" );
  }

  const std::string vertexShaderSrc = util::ReadTextFile( "shader.vert" );
  const std::string fragentShaderSrc = util::ReadTextFile( "shader.frag" );

  mVertexShader = CreateShader( GL_VERTEX_SHADER, vertexShaderSrc );
  mFragmentxShader = CreateShader( GL_FRAGMENT_SHADER, fragentShaderSrc );

  glLinkProgram( mShaderProgram );
  int32_t linked( GL_FALSE );
  glGetProgramiv( mShaderProgram, GL_LINK_STATUS, &linked );
  if ( linked == GL_FALSE )
  {
    int32_t info_size( 0 );
    glGetProgramiv( mShaderProgram, GL_INFO_LOG_LENGTH, &info_size );
    std::string msg;
    if ( info_size > 0 )
    {
      std::string buffer( info_size++, ' ' );
      glGetProgramInfoLog( mShaderProgram, info_size, NULL, &buffer[0] );
      msg.swap( buffer );
    }
    std::stringstream ss;
    ss << "cannot link shader program: " << msg;
    throw std::runtime_error( ss.str() );
  }
}

uint32_t GLCanvas::CreateShader( uint32_t kind, const std::string& src )
{
  uint32_t shaderId = glCreateShader( kind );
  if ( shaderId == 0 )
  {
    std::stringstream ss;
    ss << "cannot create shader " << kind;
    throw std::runtime_error( ss.str() );
  }

  glAttachShader( mShaderProgram, shaderId );
  const char* str[] = { src.c_str() };
  glShaderSource( shaderId, 1, str, 0 );

  glCompileShader( shaderId );

  int32_t compiled( GL_FALSE );
  glGetShaderiv( shaderId, GL_COMPILE_STATUS, &compiled );
  if ( compiled == GL_FALSE )
  {
    int32_t info_size( 0 );

    glGetShaderiv( shaderId, GL_INFO_LOG_LENGTH, &info_size );
    std::string msg;
    if ( info_size > 0 )
    {
      std::string buffer( info_size++, ' ' );
      glGetShaderInfoLog( shaderId, info_size, NULL, &buffer[0] );
      msg.swap( buffer );
    }
    std::stringstream ss;
    ss << "cannot compile shader " << kind << ". msg : " << msg;
    throw std::runtime_error( ss.str() );
  }

  return shaderId;
}

void GLCanvas::CreateTextures()
{
  // create world texture
  {
    mTextures.push_back( std::make_unique<Texture>( mTextureSize, mTextureSize ) );

    const size_t pixelCount = mTextures.front()->Width() * static_cast<size_t>( mTextures.front()->Height() );
    const size_t byteCount = pixelCount * sizeof( uint32_t );

    // allocate PBO pixels
    mPBOs[mBackBufferIdx]->BindPbo();
    mPBOs[mBackBufferIdx]->Allocate( byteCount );
    mPBOs[mBackBufferIdx]->RegisterCudaResource(); // TODO check this
    mPBOs[mBackBufferIdx]->UnbindPbo();

    mPBOs[mFrontBufferIdx]->BindPbo();
    mPBOs[mFrontBufferIdx]->Allocate( byteCount );
    mPBOs[mFrontBufferIdx]->RegisterCudaResource(); // TODO check this

    // initialize front buffer
    uint32_t* devicePixelBufferPtr = mPBOs[mFrontBufferIdx]->MapPboBuffer(); // TODO no raw ptr
    std::fill( devicePixelBufferPtr, devicePixelBufferPtr + pixelCount, mSecondaryColor );
    mPBOs[mFrontBufferIdx]->UnmapPboBuffer();

    // update texture from front buffer
    mTextures.back()->Bind();
    mTextures.back()->CreateFromPBO();
    mTextures.back()->Unbind();
    mPBOs[mFrontBufferIdx]->UnbindPbo();

    std::stringstream ss;
    ss << " World created with dimensions: " << mTextures.back()->Width() << "x" << mTextures.back()->Height();
    dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  }

  // create pixel grid checkerboard texture
  {
    mTextures.push_back( std::make_unique<Texture>( mTexturePatternSize, mTexturePatternSize, GL_REPEAT ) );

    const uint32_t pixelCount = mTextures.back()->Width() * mTextures.back()->Height();
    std::unique_ptr<uint32_t[]> pixelBuffer = std::make_unique<uint32_t[]>( pixelCount );

    // TODO do on GPU instead ?
    const uint8_t alpha = 100;
    const uint32_t backgroundColor = util::Color( 0, 0, 0, 0 );
    const uint32_t darkForegroundColor = util::Color( 130, 130, 130, alpha );
    const uint32_t lightForegroundColor = util::Color( 205, 205, 205, alpha );
    for ( uint32_t j = 0; j < mTextures.back()->Height(); ++j )
    {
      for ( uint32_t i = 0; i < mTextures.back()->Width(); ++i )
      {
        const bool isGridPixel = ( ( ( i & 0x1 ) == 0 ) ^ ( ( j & 0x1 ) == 0 ) );
        uint32_t color = isGridPixel ? darkForegroundColor : backgroundColor;
        if ( isGridPixel && ( (i % 10 == 0) || (j % 10 == 0) ) )
        {
          color = lightForegroundColor;
        }

        const uint32_t offset = i + j * mTextures.back()->Width();
        pixelBuffer[offset] = color;
      }
    }

    mTextures.back()->Bind();
    mTextures.back()->CreateFromArray( pixelBuffer.get() );
    mTextures.back()->Unbind();
    std::stringstream ss;
    ss << " Checkerboard texture created with dimensions: " << mTextures.back()->Width() << "x" << mTextures.back()->Height();
    dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  }

  // create pattern textures
  {
    for ( const auto& patternInfo : mDrawPatterns )
    {
      mTextures.push_back( std::make_unique<Texture>( patternInfo->mPattern->Width(), patternInfo->mPattern->Height() ) );
      const uint32_t pixelCount = mTextures.back()->Width() * mTextures.back()->Height();
      std::unique_ptr<uint32_t[]> pixelBuffer = std::make_unique<uint32_t[]>( pixelCount );

      const uint8_t alpha = 155;
      const uint32_t backgroundColor = util::Color( 0, 0, 0, 0 );       // TODO: use current color instead in shader
      const uint32_t foregroundColor = util::Color( 200, 200, 200, alpha ); // TODO: use current color instead in shader
      for ( uint32_t j = 0; j < mTextures.back()->Height(); ++j )
      {
        for ( uint32_t i = 0; i < mTextures.back()->Width(); ++i )
        {
          pixelBuffer[i + j * mTextures.back()->Width()] = patternInfo->mPattern->At( i, j ) ? foregroundColor : backgroundColor;
        }
      }
      mTextures.back()->Bind();
      mTextures.back()->CreateFromArray( pixelBuffer.get() );
      mTextures.back()->Unbind();

      patternInfo->mTextureIdx = mTextures.size() - 1;

      std::stringstream ss;
      ss << " Pattern created with dimensions: " <<mTextures.back()->Width() << "x" << mTextures.back()->Height();
      dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
    }
  }
}

void GLCanvas::CreatePatterns()
{
  if ( std::filesystem::exists( "e:\\patterns.dat" ) )
  {
    std::ifstream patternsStream( "e:\\patterns.dat" );
    size_t count = 0;
    util::Read_t( patternsStream, count );
    mDrawPatterns.resize( count );
    for( size_t i(0); i < count; ++i )
    {
      mDrawPatterns[i] = std::make_unique<PatternInfo>( PatternInfo( std::make_unique<Pattern>() ) );
      mDrawPatterns[i]->mPattern->Read( patternsStream );
    }
  }
  else
  {
    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Pixel", 1, 1, std::vector<bool> {
      1 } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Block", 2, 2, std::vector<bool> {
      1, 1,
      1, 1 } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "BigBlock", 5, 5, std::vector<bool> {
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 0, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1 } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Glider", 3, 3, std::vector<bool> {
      0, 1, 0,
        0, 0, 1,
        1, 1, 1 } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Eater", 4, 4, std::vector<bool> {
        1, 1, 0, 0,
        1, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 1 } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Lightweight Spaceship", 5, 4, std::vector<bool> {
      0, 0, 1, 1, 0,
        1, 1, 0, 1, 1,
        1, 1, 1, 1, 0,
        0, 1, 1, 0, 0 } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Middleweight Spaceship", 6, 4, std::vector<bool> {
      0, 0, 0, 1, 1, 0,
        1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 0, 0 } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Heavyweight Spaceship", 7, 4, std::vector<bool> {
      0, 0, 0, 0, 1, 1, 0,
        1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0 } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Glider Gun", 36, 9, std::vector<bool> {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, } ) ) ) );

    mDrawPatterns.push_back( std::make_unique<PatternInfo>( std::move( std::make_unique<Pattern>( "Unknown Ship", 34, 35, std::vector<bool> {
      0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,1,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,0,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,1,0,0,1,1,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,1,0,0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,0,0,1,0,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,1,0,0,0,1,1,1,1,1,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0 } ) ) ) );
  }
}

math::vec2 GLCanvas::ScreenToWorld( const math::ivec2& screenSpacePoint )
{
  const math::vec4 ndc( screenSpacePoint.x / static_cast<float>( GetSize().GetX() ) * 2.0f - 1.0f
                        , -screenSpacePoint.y / static_cast<float>( GetSize().GetY() ) * 2.0f + 1.0f
                        , 0.0f, 1.0f );

  const math::mat4 invVpMatrix( glm::inverse( mProjectionMatrix * mViewMatrix ) );
  const math::vec4 worldSpacePoint( invVpMatrix * ndc ); // !!  vector and column matrix multiplication !!
  return math::vec2( worldSpacePoint );
}

math::ivec2 GLCanvas::WorldToImage( const math::vec2& worldSpacePoint )
{
  const float x( worldSpacePoint.x / mQuadSize * mTextures.front()->Width() );
  const float y( -worldSpacePoint.y / mQuadSize * mTextures.front()->Height() + mTextures.front()->Height() );  // texture`s and world`s y are in the opposite order
  if ( x < 0.0f || x >= static_cast<float>( mTextures.front()->Width() ) || y < 0.0f || y >= static_cast<float>( mTextures.front()->Height() ) )
  {
    return math::ivec2( -1, -1 );
  }
  return math::ivec2( glm::floor( x ), glm::floor( y ) );
}

void GLCanvas::SetPixel( const math::uvec2& pixel )
{
  try
  {
    // calculate updateable pixel region coordinates
    const int32_t tox = pixel.x ;//- ( mDrawPatterns[mDrawPatternIdx]->mPattern->Width() / 2.0f ) + 0.5f;
    const int32_t toy = pixel.y ;//- ( mDrawPatterns[mDrawPatternIdx]->mPattern->Height() / 2.0f ) + 0.5f;
    const uint32_t pw = mDrawPatterns[mDrawPatternIdx]->mPattern->Width();
    const uint32_t ph = mDrawPatterns[mDrawPatternIdx]->mPattern->Height();

    const uint32_t rxmin = std::max( tox, 0 );
    const uint32_t rymin = std::max( toy, 0 );
    const uint32_t rxmax = std::min( tox + mDrawPatterns[mDrawPatternIdx]->mPattern->Width(), mTextures.front()->Width() );
    const uint32_t rymax = std::min( toy + mDrawPatterns[mDrawPatternIdx]->mPattern->Height(), mTextures.front()->Height() );

    uint32_t pox = 0;
    uint32_t poy = 0;
    if ( tox < 0 )
    {
      pox -= tox;
    }
    if ( toy < 0 )
    {
      poy -= toy;
    }

    //{
    //  std::stringstream ss;
    //  ss << "SetPixel rxmin:" << rxmin << " rxmax: " << rxmax << " rymin: " << rymin << " rymax: " << rymax;
    //  dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
    //}

    // update pixels in front PBO
    // y coordinates are from top to down in screen space
    mPBOs[mFrontBufferIdx]->BindPbo();
    uint32_t* pixelBuffer = mPBOs[mFrontBufferIdx]->MapPboBuffer();

    uint32_t pcx = pox;
    uint32_t pcy = poy;
    for ( uint32_t py = rymin; py < rymax; ++py )
    {
      for ( uint32_t px = rxmin; px < rxmax; ++px )
      {
        if ( mDrawPatterns[mDrawPatternIdx]->mPattern->At( pcx, pcy ) )
        {
          pixelBuffer[px + py * mTextures.front()->Width() ] = mCurrentDrawingColor; 
        }
        ++pcx;
      }
      ++pcy;
      pcx = pox;
    }

    mPBOs[mFrontBufferIdx]->UnmapPboBuffer();

    // update texture region from front PBO
    mTextures.front()->Bind();
    mTextures.front()->UpdateFromPBO( rxmin, rymin, rxmax - rxmin, rymax - rymin );
    mTextures.front()->Unbind();
    mPBOs[mFrontBufferIdx]->UnbindPbo();
  }
  catch ( const std::exception& e )
  {
    std::stringstream ss;
    ss << "SetPixel error: " << e.what();
    dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  }
}

void GLCanvas::Reset()
{
  // reset pixels in the front buffer
  mPBOs[mFrontBufferIdx]->MapCudaResource();
  uint32_t* mappedPtr = mPBOs[mFrontBufferIdx]->GetCudaMappedPointer();
  RunFillKernel( mappedPtr, mSecondaryColor, mTextures.front()->Width(), mTextures.front()->Height() );
  mPBOs[mFrontBufferIdx]->UnmapCudaResource();

  // update texture from the front buffer
  mPBOs[mFrontBufferIdx]->BindPbo();
  mTextures.front()->Bind();
  mTextures.front()->UpdateFromPBO();
  mTextures.front()->Bind();
  mPBOs[mFrontBufferIdx]->UnbindPbo();

  Refresh();
}

void GLCanvas::Random()
{
  // reset pixels in the front buffer
  mPBOs[mFrontBufferIdx]->MapCudaResource();
  uint32_t* mappedPtr = mPBOs[mFrontBufferIdx]->GetCudaMappedPointer();

  RunRandomKernel( mappedPtr, 0.9f, mTextures.front()->Width(), mTextures.front()->Height(), mPrimaryColor, mSecondaryColor );

  mPBOs[mFrontBufferIdx]->UnmapCudaResource();

  // update texture from the front buffer
  mPBOs[mFrontBufferIdx]->BindPbo();
  mTextures.front()->Bind();
  mTextures.front()->UpdateFromPBO();
  mTextures.front()->Unbind();
  mPBOs[mFrontBufferIdx]->UnbindPbo();

  Refresh();
}

void GLCanvas::RotatePattern()
{
  //mDrawPatterns[mDrawPatternIdx]->mPattern->Rotate();
}

void GLCanvas::Step()
{
  mPBOs[mFrontBufferIdx]->MapCudaResource();
  mPBOs[mBackBufferIdx]->MapCudaResource();
  uint32_t* mappedFrontPtr = mPBOs[mFrontBufferIdx]->GetCudaMappedPointer();
  uint32_t* mappedBackPtr = mPBOs[mBackBufferIdx]->GetCudaMappedPointer();
  RunStepKernel( mappedFrontPtr, mappedBackPtr, mTextures.front()->Width(), mTextures.front()->Height(), mPrimaryColor, mSecondaryColor );
  mPBOs[mFrontBufferIdx]->UnmapCudaResource();
  mPBOs[mBackBufferIdx]->UnmapCudaResource();

  std::swap( mFrontBufferIdx, mBackBufferIdx );

  mPBOs[mFrontBufferIdx]->BindPbo();
  mTextures.front()->Bind();
  mTextures.front()->UpdateFromPBO();
  mTextures.front()->Unbind();
  mPBOs[mFrontBufferIdx]->UnbindPbo();

  Refresh();
}

void GLCanvas::Start()
{
  if ( mStepTimerRuning )
  {
    return;
  }

  mStepTimer->Start( mStepDeltaTime );
  mStepTimerRuning = true;
}

void GLCanvas::Stop()
{
  if ( !mStepTimerRuning )
  {
    return;
  }

  mStepTimer->Stop();
  mStepTimerRuning = false;
}

void GLCanvas::SetDeltaTime( const uint32_t dt )
{
  mStepDeltaTime = dt;
  if ( mStepTimerRuning )
  {
    Stop();
    Start();
  }
}

void GLCanvas::OnPaint( wxPaintEvent& /*event*/ )
{
  SetCurrent( *mContext );

  glClearColor( 0.1f, 0.1f, 0.1f, 1.0f );
  glEnable( GL_BLEND );
  glBlendEquation( GL_FUNC_ADD );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glClear( GL_COLOR_BUFFER_BIT );

  glUseProgram( mShaderProgram );

  // draw world
  {
    const math::mat4 vpMatrix( mProjectionMatrix * mViewMatrix );
    const int32_t uniformLoc( glGetUniformLocation( mShaderProgram, "vpMatrix" ) ); // TODO do it nicer inside a shader object
    glUniformMatrix4fv( uniformLoc, 1, GL_FALSE, &vpMatrix[0][0] );

    mMeshes[0]->Bind();
    mTextures[0]->BindTextureUnit( 0 ); // TODO should the Mesh do this instead in it's Render method?

    glUniform1i( glGetUniformLocation( mShaderProgram, "textureData" ), 0 ); // TODO textureData at location 0. Should the Mesh do this instead

    mMeshes[0]->Render();

    mTextures[0]->UnbindTextureUnit();
    mMeshes[0]->Unbind();
  }

  // draw pixel grid if needed
  if ( mDrawPixelGrid )
  {
    glUseProgram( mShaderProgram );

    const math::mat4 vpMatrix( mProjectionMatrix * mViewMatrix );
    const int32_t uniformLoc( glGetUniformLocation( mShaderProgram, "vpMatrix" ) ); // TODO do it nicer inside a shader object
    glUniformMatrix4fv( uniformLoc, 1, GL_FALSE, &vpMatrix[0][0] );

    mMeshes[1]->Bind();
    mTextures[1]->BindTextureUnit( 0 ); // TODO should the Mesh do this instead in it's Render method?

    glUniform1i( glGetUniformLocation( mShaderProgram, "textureData" ), 0 ); // textureData at location 0. Should the Mesh do this instead in it's Render method?

    mMeshes[1]->Render();

    mTextures[1]->UnbindTextureUnit();
    mMeshes[1]->Unbind();

    // draw current pattern
    // calculate the size of a pixel with the current view. if bigger than a threshold then draw the pattern.
    if ( mViewMatrix[0][0] > 1.0f )
    {
      const math::vec2 pixelSize( mQuadSize / mTextures[0]->Width(), mQuadSize / mTextures[0]->Height() );
      const math::vec2 position( mCurrentMouseWorldPosition );
      const math::vec2 roundedPosition( util::RoundToNearestMultiple( position.x, pixelSize.x ), util::RoundToNearestMultiple( position.y, pixelSize.y ) + pixelSize.y );
      math::mat4 modelMatrix = glm::translate( glm::identity<math::mat4>(), math::vec3( roundedPosition, 0.0 ) ); // roundedPosition

      const math::mat4 mvpMatrix = mProjectionMatrix * mViewMatrix * modelMatrix;  // TODO model matrix will be needed to rotate the pattern quad!
      const int32_t uniformLoc( glGetUniformLocation( mShaderProgram, "vpMatrix" ) ); // TODO do it nicer inside a shader object
      glUniformMatrix4fv( uniformLoc, 1, GL_FALSE, &mvpMatrix[0][0] );

      mMeshes[mDrawPatterns[mDrawPatternIdx]->mMeshIdx]->Bind();
      mTextures[mDrawPatterns[mDrawPatternIdx]->mTextureIdx]->BindTextureUnit( 0 ); // TODO should the Mesh do this instead in it's Render method?

      glUniform1i( glGetUniformLocation( mShaderProgram, "textureData" ), 0 ); // textureData at location 0. Should the Mesh do this instead in it's Render method?

      mMeshes[mDrawPatterns[mDrawPatternIdx]->mMeshIdx]->Render();

      mTextures[mDrawPatterns[mDrawPatternIdx]->mTextureIdx]->UnbindTextureUnit();
      mMeshes[mDrawPatterns[mDrawPatternIdx]->mMeshIdx]->Unbind();
    }
  }

  glUseProgram( 0 );

  SwapBuffers();
}

void GLCanvas::OnSize( wxSizeEvent& event )
{
  glViewport( 0, 0, event.GetSize().GetX(), event.GetSize().GetY() );

  const float aspectRatio = static_cast<float>( event.GetSize().GetX() ) / static_cast<float>( event.GetSize().GetY() );
  float xSpan = 1.0f;
  float ySpan = 1.0f;

  if ( aspectRatio > 1.0f )
  {
    xSpan *= aspectRatio;
  }
  else
  {
    ySpan = xSpan / aspectRatio;
  }

  mProjectionMatrix = glm::ortho( -1.0f * xSpan, xSpan, -1.0f * ySpan, ySpan );
}

void GLCanvas::OnMouseMove( wxMouseEvent& event )
{
  const math::ivec2 screenPos( event.GetX(), event.GetY() );
  const math::vec2 worldPos( ScreenToWorld( screenPos ) );
  const math::ivec2 imagePos( WorldToImage( worldPos ) );

  if ( mPanningActive )
  {
    const math::vec2 mouse_delta( worldPos - ScreenToWorld( mPreviousMousePosition ) );

    mViewMatrix = glm::translate( mViewMatrix, math::vec3( mouse_delta, 0.0f ) );
    mPreviousMousePosition = screenPos;
    //if ( !mStepTimerRuning )
    {
      Refresh();
    }
  }
  else if ( mDrawingActive && imagePos != math::ivec2( -1, -1 ) )
  {
    SetPixel( imagePos );
    //if ( !mStepTimerRuning )
    {
      Refresh();
    }
  }
  
  mCurrentMouseWorldPosition = worldPos;
  if ( !mStepTimerRuning )
  {
    Refresh();
  }
  //{
  //  std::stringstream ss;
  //  ss << "screen: " << screenPos << " world: " << worldPos;
  //  if ( imagePos != math::ivec2( -1, -1 ) )
  //  {
  //    ss << " image: " << imagePos;
  //  }
  //}
}

void GLCanvas::OnMouseWheel( wxMouseEvent& event )
{
  const float scaleFactor( 0.1f );
  const float scale( event.GetWheelRotation() < 0 ? 1.0f - scaleFactor : 1.0f + scaleFactor );

  const math::vec2 focusPoint( static_cast<float>( event.GetX() ), static_cast<float>( event.GetY() ) );
  const math::vec2 worldFocusPoint = ScreenToWorld( focusPoint );

  mViewMatrix = glm::translate( glm::scale( glm::translate( mViewMatrix
                                                            , math::vec3( worldFocusPoint, 0.0f ) )
                                            , math::vec3( scale, scale, 1.0f ) )
                , math::vec3( -worldFocusPoint, 0.0f ) );

  Refresh();
}

void GLCanvas::OnMouseRightDown( wxMouseEvent& event )
{
  const math::ivec2 screenPos( event.GetX(), event.GetY() );
  const math::vec2 worldPos( ScreenToWorld( screenPos ) );
  const math::ivec2 imagePos( WorldToImage( worldPos ) );

  if ( imagePos == math::ivec2( -1, -1 ) )
  {
    return;
  }

  mDrawingActive = true;
  mCurrentDrawingColor = mSecondaryColor;
  SetPixel( imagePos );
  Refresh();

  this->SetFocus();
}

void GLCanvas::OnMouseRightUp( wxMouseEvent& /*event*/ )
{
  mDrawingActive = false;
}

void GLCanvas::OnMouseMiddleDown( wxMouseEvent& event )
{
  mPreviousMousePosition = math::vec2( static_cast<float>( event.GetX() ), static_cast<float>( event.GetY() ) );
  mPanningActive = true;
}

void GLCanvas::OnMouseMiddleUp( wxMouseEvent& /*event*/ )
{
  mPanningActive = false;
}

void GLCanvas::OnMouseLeftDown( wxMouseEvent& event )
{
  const math::ivec2 screenPos( event.GetX(), event.GetY() );
  const math::vec2 worldPos( ScreenToWorld( screenPos ) );
  const math::ivec2 imagePos( WorldToImage( worldPos ) );

  if ( imagePos == math::ivec2( -1, -1 ) )
  {
    return;
  }

  mDrawingActive = true;
  mCurrentDrawingColor = mPrimaryColor;
  SetPixel( imagePos );
  Refresh();

  this->SetFocus();
}

void GLCanvas::OnMouseLeftUp( wxMouseEvent& /*event*/ )
{
  mDrawingActive = false;
}

void GLCanvas::OnMouseLeave( wxMouseEvent& /*event*/ )
{
  mPanningActive = mDrawingActive = false;
}

void GLCanvas::OnKeyDown( wxKeyEvent& event )
{
  if ( event.GetKeyCode() == 'R' )
  {
    RotatePattern();
  }
  else if ( event.GetKeyCode() == WXK_SPACE )
  {
    if ( mStepTimerRuning )
    {
      Stop();
    }
    else
    {
      Start();
    }
  }
  else
  {
    event.Skip();
  }
}

void GLCanvas::OnStepTimer()
{
  try
  {
    Step();
  }
  catch ( const std::exception& )
  {
    mStepTimer->Stop();
  }
  catch ( ... )
  {
    mStepTimer->Stop();
  }
}

// StepTimer
GLCanvas::StepTimer::StepTimer( GLCanvas* parent )
  : mParent( parent )
{}

void GLCanvas::StepTimer::Notify()
{
  mParent->OnStepTimer();
}