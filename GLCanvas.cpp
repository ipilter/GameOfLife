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
{
  wxGLContextAttrs contextAttrs;
  contextAttrs.CoreProfile().OGLVersion( 4, 5 ).Robust().ResetIsolation().EndList();
  mContext = std::make_unique<wxGLContext>( this, nullptr, &contextAttrs );

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

  SetCurrent( *mContext );
  InitializeGLEW();

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

  // create PBOs
  mPBOs.push_back( std::make_unique<PixelBufferObject>() );
  mPBOs.push_back( std::make_unique<PixelBufferObject>() );
  mFrontBufferIdx = 0;
  mBackBufferIdx = 1;

  // stuff
  int32_t maxTextureSize = 0;
  glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTextureSize );
  logger::Logger::instance() << "Max texture size on current GPU " << maxTextureSize << "x" << maxTextureSize << "\n";
  mTextureSize = glm::min( mTextureSize, static_cast<uint32_t>( maxTextureSize ) ); // nxn pixels
  mTexturePatternSize = 10; // nxn pixels

  CreateGeometry();
  CreateShaderProgram();
  CreateTextures();

  // view
  mViewMatrix = glm::translate( glm::scale( glm::identity<math::mat4>(), math::vec3( 5.0f ) ), math::vec3( -mQuadSize / 2.0, -mQuadSize / 2.0, 0.0 ) );

  // patterns
  if ( std::filesystem::exists( "e:\\patterns.dat" ) )
  {
    std::ifstream patternsStream( "e:\\patterns.dat" );
    size_t count = 0;
    util::Read_t( patternsStream, count );
    mDrawPatterns.resize( count );
    for( auto i(0); i < count; ++i )
    {
      mDrawPatterns[i] = std::move( std::make_unique<Pattern>() );
      mDrawPatterns[i]->read( patternsStream );
    }
  }
  else
  {
    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( "Pixel", 1, 1, std::vector<bool> {
                                                                      1 } ) ) );

    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( "Glider", 3, 3, std::vector<bool> {
                                                                      0, 1, 0,
                                                                      0, 0, 1,
                                                                      1, 1, 1 } ) ) );

    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( "Lightweight Spaceship", 5, 4, std::vector<bool> {
                                                                      0, 0, 1, 1, 0,
                                                                      1, 1, 0, 1, 1,
                                                                      1, 1, 1, 1, 0,
                                                                      0, 1, 1, 0, 0 } ) ) );

    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( "Middleweight Spaceship", 6, 4, std::vector<bool> {
                                                                      0, 0, 0, 1, 1, 0,
                                                                      1, 1, 1, 0, 1, 1,
                                                                      1, 1, 1, 1, 1, 0,
                                                                      0, 1, 1, 1, 0, 0 } ) ) );

    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( "Heavyweight Spaceship", 7, 4, std::vector<bool> {
                                                                      0, 0, 0, 0, 1, 1, 0,
                                                                      1, 1, 1, 1, 0, 1, 1,
                                                                      1, 1, 1, 1, 1, 1, 0,
                                                                      0, 1, 1, 1, 1, 0, 0 } ) ) );

    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( "Glider Gun", 36, 9, std::vector<bool> {
                                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                                                      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, } ) ) );

    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( "Unknown Ship", 34, 35, std::vector<bool> {
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
                                                                      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0 } ) ) );
  }

  mStepTimer = std::make_unique<StepTimer>( this );
}

GLCanvas::~GLCanvas()
{
  {
    std::ofstream patternsStream( "e:\\patterns.dat" );
    util::Write_t( patternsStream, mDrawPatterns.size() );
    for ( auto& p : mDrawPatterns )
    {
      p->write( patternsStream );
    }
  }

  SetCurrent( *mContext );

  glDeleteShader( mVertexShader );
  glDeleteShader( mFragmentxShader );
  glDeleteProgram( mShaderProgram );

  glDeleteVertexArrays( 1, &mVao );
  glDeleteBuffers( 1, &mVbo );
  glDeleteBuffers( 1, &mIbo );
}

void GLCanvas::SetPrimaryColor( const math::uvec3& color )
{
  mPrimaryColor = color;
}

const math::uvec3& GLCanvas::GetPrimaryColor() const
{
  return mPrimaryColor;
}

void GLCanvas::SetSecondaryColor( const math::uvec3& color )
{
  mSecondaryColor = color;
}

const math::uvec3& GLCanvas::GetSecondaryColor() const
{
  return mSecondaryColor;
}

void GLCanvas::SetCurrentPattern( const uint32_t idx )
{
  mDrawPattern = *mDrawPatterns[idx];
}

uint32_t GLCanvas::GetPatternCount() const
{
  return mDrawPatterns.size();
}

const Pattern& GLCanvas::GetPattern( const uint32_t idx ) const
{
  return *mDrawPatterns[idx];
}

void GLCanvas::SetDrawPixelGrid( const bool drawPixelGrid )
{
  mDrawPixelGrid = drawPixelGrid;

  Refresh();
}

void GLCanvas::InitializeGLEW()
{
  glewExperimental = false;
  GLenum err = glewInit();
  if ( err != GLEW_OK )
  {
    const GLubyte* msg = glewGetErrorString( err );
    throw std::exception( reinterpret_cast<const char*>( msg ) );
  }

  auto fp64 = glewGetExtension( "GL_ARB_gpu_shader_fp64" );
  logger::Logger::instance() << "GL_ARB_gpu_shader_fp64 " << ( fp64 == 1 ? "supported" : "not supported" ) << "\n";
}

void GLCanvas::CreateGeometry()
{
  const float pixelPatternTexelSize = mTextureSize / static_cast<float>( mTexturePatternSize );
  const std::vector<float> points = { 0.0f,      0.0f,       0.0f, 1.0f, 0.0,                   pixelPatternTexelSize // vtx bl
                                     , mQuadSize, 0.0f,      1.0f, 1.0f, pixelPatternTexelSize, pixelPatternTexelSize // vtx br
                                     , 0.0f,      mQuadSize, 0.0f, 0.0f, 0.0,                   0.0 // vtx tl
                                     , mQuadSize, mQuadSize, 1.0f, 0.0f, pixelPatternTexelSize, 0.0 // vtx tr
  };

  const std::vector<uint32_t> indices = { 0, 1, 2,  1, 3, 2 };  // triangle vertex indices

  glGenBuffers( 1, &mVbo );
  glBindBuffer( GL_ARRAY_BUFFER, mVbo );
  glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * points.size(), &points.front(), GL_STATIC_DRAW );

  glGenBuffers( 1, &mIbo );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mIbo );
  glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( uint32_t ) * indices.size(), &indices.front(), GL_STATIC_DRAW );

  glGenVertexArrays( 1, &mVao );
  glBindVertexArray( mVao );
  glEnableVertexAttribArray( 0 );
  glEnableVertexAttribArray( 1 );
  glEnableVertexAttribArray( 2 );
  glBindBuffer( GL_ARRAY_BUFFER, mVbo );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mIbo );

  uint32_t stride = 6 * sizeof( float );
  size_t vertexOffset = 0;
  size_t worldTexelOffset = 2 * sizeof( float );
  size_t patternTexelOffset = 4 * sizeof( float );
  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( vertexOffset ) );
  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( worldTexelOffset ) );
  glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( patternTexelOffset ) );
  glBindVertexArray( 0 );
}

void GLCanvas::CreateShaderProgram()
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

    const auto pixelCount = mTextures.front()->width() * mTextures.front()->height();
    const auto byteCount = pixelCount * 4ull;

    // allocate PBO pixels
    mPBOs[mBackBufferIdx]->bindPbo();
    mPBOs[mBackBufferIdx]->allocate( byteCount );
    mPBOs[mBackBufferIdx]->registerCudaResource(); // TODO check this
    mPBOs[mBackBufferIdx]->unbindPbo();

    mPBOs[mFrontBufferIdx]->bindPbo();
    mPBOs[mFrontBufferIdx]->allocate( byteCount );
    mPBOs[mFrontBufferIdx]->registerCudaResource(); // TODO check this

    // initialize front buffer
    uint8_t* pixelBuffer = mPBOs[mFrontBufferIdx]->mapPboBuffer();
    std::fill( pixelBuffer, pixelBuffer + byteCount, 0 );
    mPBOs[mFrontBufferIdx]->unmapPboBuffer();

    // update texture from front buffer
    mTextures.back()->bind();
    mTextures.back()->createFromPBO();
    mTextures.back()->unbind();
    mPBOs[mFrontBufferIdx]->unbindPbo();
  }

  // create pixel grid checkerboard texture
  {
    mTextures.push_back( std::make_unique<Texture>( mTexturePatternSize, mTexturePatternSize, GL_REPEAT ) );

    const uint8_t backgroundColor = 20;
    const uint8_t darkForegroundColor = 30;
    const uint8_t lightForegroundColor = 50;
    const auto pixelCount = mTextures.back()->width() * mTextures.back()->height();
    const auto byteCount = pixelCount * 4ull; // TODO no need for RGBA here

    std::unique_ptr<uint8_t[]> pixelBuffer = std::make_unique<uint8_t[]>( byteCount );
    for ( uint32_t j = 0; j < mTextures.back()->height(); ++j )
    {
      for ( uint32_t i = 0; i < mTextures.back()->width(); ++i )
      {
        const bool isGridPixel = ( ( ( i & 0x1 ) == 0 ) ^ ( ( j & 0x1 ) == 0 ) );
        uint8_t color = isGridPixel ? darkForegroundColor : backgroundColor;
        if ( isGridPixel && ( (i % 10 == 0) || (j % 10 == 0) ) )
        {
          color = lightForegroundColor;
        }

        const auto offset = i * 4ull + j * 4ull * mTextures.back()->width();
        pixelBuffer[offset + 0] = color;
        pixelBuffer[offset + 1] = color;
        pixelBuffer[offset + 2] = color;
        pixelBuffer[offset + 3] = 0;
      }
    }

    mTextures.back()->bind();
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, mTextures.back()->width(), mTextures.back()->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelBuffer.get() );
    mTextures.back()->unbind();
  }

  std::stringstream ss;
  ss << " Texture with dimensions " << mTextures.front()->width() << "x" << mTextures.front()->height() << " created";
  dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  logger::Logger::instance() << "Creating texture with size " << mTextures.front()->width() << "x" << mTextures.front()->height() << "\n";
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
  const float x( worldSpacePoint.x / mQuadSize * mTextures.front()->width() );
  const float y( -worldSpacePoint.y / mQuadSize * mTextures.front()->height() + mTextures.front()->height() );  // texture`s and world`s y are in the opposite order
  if ( x < 0.0f || x >= static_cast<float>( mTextures.front()->width() ) || y < 0.0f || y >= static_cast<float>( mTextures.front()->height() ) )
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
    int32_t tox = pixel.x - ( mDrawPattern.width() / 2.0f ) + 0.5f;
    int32_t toy = pixel.y - ( mDrawPattern.height() / 2.0f ) + 0.5f;
    uint32_t pw = mDrawPattern.width();
    uint32_t ph = mDrawPattern.height();

    uint32_t rxmin = std::max( tox, 0 );
    uint32_t rymin = std::max( toy, 0 );
    uint32_t rxmax = std::min( tox + mDrawPattern.width(), mTextures.front()->width() );
    uint32_t rymax = std::min( toy + mDrawPattern.height(), mTextures.front()->height() );

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

    // update pixels in front PBO
    mPBOs[mFrontBufferIdx]->bindPbo();
    uint8_t* pixelBuffer = mPBOs[mFrontBufferIdx]->mapPboBuffer();

    uint32_t pcx = pox;
    uint32_t pcy = poy;
    for ( uint32_t py = rymin; py < rymax; ++py )
    {
      for ( uint32_t px = rxmin; px < rxmax; ++px )
      {
        const uint64_t offset = px * 4ull + py * 4ull * mTextures.front()->width();
        if ( mDrawPattern.at( pcx, pcy ) )
        {
          pixelBuffer[offset + 0] = mCurrentDrawingColor.x;
          pixelBuffer[offset + 1] = mCurrentDrawingColor.y;
          pixelBuffer[offset + 2] = mCurrentDrawingColor.z;
        }
        ++pcx;
      }
      ++pcy;
      pcx = pox;
    }

    mPBOs[mFrontBufferIdx]->unmapPboBuffer();

    // update texture region from front PBO
    mTextures.front()->bind();
    mTextures.front()->updateFromPBO( rxmin, rymin, rxmax - rxmin, rymax - rymin );
    mTextures.front()->unbind();
    mPBOs[mFrontBufferIdx]->unbindPbo();
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
  mPBOs[mFrontBufferIdx]->mapCudaResource();
  uint8_t* mappedPtr = mPBOs[mFrontBufferIdx]->getCudaMappedPointer();
  RunFillKernel( mappedPtr, mSecondaryColor.x, mTextures.front()->width(), mTextures.front()->height() );
  mPBOs[mFrontBufferIdx]->unmapCudaResource();

  // update texture from the front buffer
  mPBOs[mFrontBufferIdx]->bindPbo();
  mTextures.front()->bind();
  mTextures.front()->updateFromPBO();
  mTextures.front()->bind();
  mPBOs[mFrontBufferIdx]->unbindPbo();

  Refresh();
}

void GLCanvas::Random()
{
  // reset pixels in the front buffer
  mPBOs[mFrontBufferIdx]->mapCudaResource();
  uint8_t* mappedPtr = mPBOs[mFrontBufferIdx]->getCudaMappedPointer();

  RunRandomKernel( mappedPtr, mPrimaryColor.x, mSecondaryColor.x, 0.9f, mTextures.front()->width(), mTextures.front()->height() );

  mPBOs[mFrontBufferIdx]->unmapCudaResource();

  // update texture from the front buffer
  mPBOs[mFrontBufferIdx]->bindPbo();
  mTextures.front()->bind();
  mTextures.front()->updateFromPBO();
  mTextures.front()->bind();
  mPBOs[mFrontBufferIdx]->unbindPbo();

  Refresh();
}

void GLCanvas::RotatePattern()
{
  mDrawPattern.rotate();
}

void GLCanvas::Step()
{
  mPBOs[mFrontBufferIdx]->mapCudaResource();
  mPBOs[mBackBufferIdx]->mapCudaResource();
  uint8_t* mappedFrontPtr = mPBOs[mFrontBufferIdx]->getCudaMappedPointer();
  uint8_t* mappedBackPtr = mPBOs[mBackBufferIdx]->getCudaMappedPointer();
  RunStepKernel( mappedFrontPtr, mappedBackPtr, mTextures.front()->width(), mTextures.front()->height() );
  mPBOs[mFrontBufferIdx]->unmapCudaResource();
  mPBOs[mBackBufferIdx]->unmapCudaResource();

  std::swap( mFrontBufferIdx, mBackBufferIdx );

  mPBOs[mFrontBufferIdx]->bindPbo();
  mTextures.front()->bind();
  mTextures.front()->updateFromPBO();
  mTextures.front()->unbind();
  mPBOs[mFrontBufferIdx]->unbindPbo();

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
  glClear( GL_COLOR_BUFFER_BIT );

  glBindVertexArray( mVao );
  glUseProgram( mShaderProgram );

  const math::mat4 vpMatrix = mProjectionMatrix * mViewMatrix;
  const int32_t uniformLoc( glGetUniformLocation( mShaderProgram, "vpMatrix" ) ); // TODO do it nicer
  glUniformMatrix4fv( uniformLoc, 1, GL_FALSE, &vpMatrix[0][0] );

  mTextures.front()->bindTextureUnit( 0 );
  mTextures.back()->bindTextureUnit( 1 );

  glUniform1i( glGetUniformLocation( mShaderProgram, "textureData" ), 0 ); // TODO do it nicer
  glUniform1i( glGetUniformLocation( mShaderProgram, "checkerboardData" ), 1 ); // TODO do it nicer
  glUniform1i( glGetUniformLocation( mShaderProgram, "isCheckerboard" ), mDrawPixelGrid ? 1 : 0 ); // TODO do it nicer

  glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );  // 6 = index count

  mTextures.front()->unbindTextureUnit();
  mTextures.back()->unbindTextureUnit();

  glBindVertexArray( 0 );
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
    Refresh();
  }
  else if ( mDrawingActive && imagePos != math::ivec2( -1, -1 ) )
  {
    SetPixel( imagePos );
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