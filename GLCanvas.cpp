#include <vector>
#include <sstream>
#include <algorithm>

#include "timer.h"
#include "GLCanvas.h"
#include "MainFrame.h"
#include "Util.h"
#include "Logger.h"
#include "kernel.cuh"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

GLCanvas::GLCanvas( const uint32_t textureExponent
                    , wxWindow* parent
                    , wxWindowID id
                    , const int32_t* attribList
                    , const wxPoint& pos
                    , const wxSize& size
                    , long style
                    , const wxString& name
                    , const wxPalette& palette )
  : wxGLCanvas( parent, id, attribList, pos, size, style, name, palette )
  , mTextureExponent( textureExponent )
{
  wxGLContextAttrs contextAttrs;
  contextAttrs.CoreProfile().OGLVersion( 4, 5 ).Robust().ResetIsolation().EndList();
  mContext = std::make_unique<wxGLContext>( this, nullptr, &contextAttrs );

  Bind( wxEVT_SIZE, &GLCanvas::OnSize, this );
  Bind( wxEVT_PAINT, &GLCanvas::OnPaint, this );
  Bind( wxEVT_RIGHT_DOWN, &GLCanvas::OnMouseRightDown, this );
  Bind( wxEVT_RIGHT_UP, &GLCanvas::OnMouseRightUp, this );
  Bind( wxEVT_LEFT_DOWN, &GLCanvas::OnMouseLeftDown, this );
  Bind( wxEVT_LEFT_UP, &GLCanvas::OnMouseLeftUp, this );
  Bind( wxEVT_MOTION, &GLCanvas::OnMouseMove, this );
  Bind( wxEVT_LEAVE_WINDOW, &GLCanvas::OnMouseLeave, this );
  Bind( wxEVT_MOUSEWHEEL, &GLCanvas::OnMouseWheel, this );

  try
  {
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
  }
  catch ( const std::exception& e )
  {
    logger::Logger::instance() << "ERROR: " << e.what() << "\n";
  }

  try
  {
    // create PBOs
    mPBOs.push_back( std::make_unique<PixelBufferObject>() );
    mPBOs.push_back( std::make_unique<PixelBufferObject>() );
    mFrontBufferIdx = 0;
    mBackBufferIdx = 1;

    CreateGeometry();
    CreateShaderProgram();
    CreateTexture();

    mViewMatrix = glm::translate( glm::scale( glm::identity<math::mat4>(), math::vec3( 1.0f ) ), math::vec3( -mQuadSize / 2.0, -mQuadSize / 2.0, 0.0 ) );

    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( 3, 3, std::vector<bool> {
                                                                     0, 1, 0,
                                                                     0, 0, 1,
                                                                     1, 1, 1 } ) ) );


    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( 5, 5, std::vector<bool> { // valid?
                                                                     0, 1, 1, 1, 1,
                                                                     1, 0, 0, 0, 1,
                                                                     0, 0, 0, 0, 1,
                                                                     1, 0, 0, 1, 0,
                                                                     0, 1, 0, 0, 0 } ) ) );

    mDrawPatterns.push_back( std::move( std::make_unique<Pattern>( 36, 9, std::vector<bool> {
                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                                                     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, } ) ) );
  }
  catch ( const std::exception& e )
  {
    logger::Logger::instance() << "ERROR: " << e.what() << "\n";
  }

  std::stringstream ss;
  ss << "GLCanvas window size: " << math::vec2( GetSize().GetWidth(), GetSize().GetHeight() );
  dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  logger::Logger::instance() << "GLCanvas window size: " << math::vec2( GetSize().GetWidth(), GetSize().GetHeight() ) << "\n";
}

GLCanvas::~GLCanvas()
{
  SetCurrent( *mContext );

  glDeleteShader( mVertexShader );
  glDeleteShader( mFragmentxShader );
  glDeleteProgram( mShaderProgram );

  glDeleteVertexArrays( 1, &mVao );
  glDeleteBuffers( 1, &mVbo );
  glDeleteBuffers( 1, &mIbo );
}

void GLCanvas::SetDrawColor( const math::uvec3& color )
{
  mDrawColor = color;
}

const math::uvec3& GLCanvas::GetDrawColor() const
{
  return mDrawColor;
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
  const std::vector<float> points = { 0.0f,      0.0f,      0.0f, 1.0f // vtx bl
                                     , mQuadSize, 0.0f,      1.0f, 1.0f // vtx br
                                     , 0.0f,      mQuadSize, 0.0f, 0.0f // vtx tl
                                     , mQuadSize, mQuadSize, 1.0f, 0.0f // vtx tr
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
  glBindBuffer( GL_ARRAY_BUFFER, mVbo );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mIbo );

  uint32_t stride = 4 * sizeof( float );
  size_t vertexOffset = 0;
  size_t texelOffset = 2 * sizeof( float );
  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( vertexOffset ) );
  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( texelOffset ) );
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

void GLCanvas::CreateTexture()
{
  // create texture
  {
    const uint32_t size = static_cast<uint32_t>( glm::pow( 2.0f, static_cast<float>( mTextureExponent ) ) );
    int32_t maxTextureSize = 0;
    glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTextureSize );
    logger::Logger::instance() << "Max texture size on current GPU " << maxTextureSize << "x" << maxTextureSize << "\n";

    mTextures.push_back( std::make_unique<Texture>( glm::min( size, static_cast<uint32_t>( maxTextureSize ) ), glm::min( size, static_cast<uint32_t>( maxTextureSize ) ) ) );
    logger::Logger::instance() << "Creating texture with size " << mTextures.front()->width() << "x" << mTextures.front()->height() << "\n";
  }

  const auto pixelCount = mTextures.front()->width() * mTextures.front()->height();
  const auto byteCount = pixelCount * 4ull;

  // allocate PBO pixels
  mPBOs[mBackBufferIdx]->bindPbo();
  mPBOs[mBackBufferIdx]->allocate( byteCount );
  mPBOs[mBackBufferIdx]->unbindPbo();

  mPBOs[mFrontBufferIdx]->bindPbo();
  mPBOs[mFrontBufferIdx]->allocate( byteCount );

  // init PBO data with zero
  {
    GLubyte* pixelBuffer = mPBOs[mFrontBufferIdx]->mapPboBuffer();
    std::fill( pixelBuffer, pixelBuffer + byteCount, 0 );
    mPBOs[mFrontBufferIdx]->unmapPboBuffer();
  }

  // bind texture
  mTextures.front()->bind();

  // create texture from PBO pixels
  mTextures.front()->createFromPBO();
  mPBOs[mFrontBufferIdx]->registerCudaResource();

  // unbind texture
  mTextures.front()->unbind();

  // unbind PBO
  mPBOs[mFrontBufferIdx]->unbindPbo();

  std::stringstream ss;
  ss << " Texture with dimensions " << mTextures.front()->width() << "x" << mTextures.front()->height() << " created";
  dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
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
    const auto& pattern = mDrawPatterns[1];

    // update pixels in PBO
    mPBOs[mFrontBufferIdx]->bindPbo();
    GLubyte* pixelBuffer = mPBOs[mFrontBufferIdx]->mapPboBuffer();
    for ( uint32_t y = 0; y < pattern->height(); ++y )
    {
      for ( uint32_t x = 0; x < pattern->width(); ++x )
      {
        const uint64_t offset = ( pixel.x + x ) * 4ull + ( pixel.y + y ) * 4ull * mTextures.front()->width();
        if ( pattern->at( x, y ) )
        {
          pixelBuffer[offset + 0] = mDrawColor.x;
          pixelBuffer[offset + 1] = mDrawColor.y;
          pixelBuffer[offset + 2] = mDrawColor.z;
        }
      }
    }
    mPBOs[mFrontBufferIdx]->unmapPboBuffer();

    // update texture region from PBO
    mTextures.front()->bind();
    mTextures.front()->updateFromPBO( pixel.x, pixel.y, pattern->width(), pattern->height() );
    mTextures.front()->unbind();

    // done
    mPBOs[mFrontBufferIdx]->unbindPbo();

    glFlush();
  }
  catch ( const std::exception& e )
  {
    std::stringstream ss;
    ss << "SetPixel error: " << e.what();
    dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  }
}

void GLCanvas::Step()
{
  Timer t;
  t.start();

  try
  {
    mPBOs[mFrontBufferIdx]->mapCudaResource();
    uint8_t* mappedPtr = mPBOs[mFrontBufferIdx]->getCudaMappedPointer();
    RunStepKernel( mappedPtr, mTextures.front()->width(), mTextures.front()->height() );
    mPBOs[mFrontBufferIdx]->unmapCudaResource();

    mPBOs[mFrontBufferIdx]->bindPbo();

    mTextures.front()->bind();
    mTextures.front()->updateFromPBO();
    mTextures.front()->unbind();
    mPBOs[mFrontBufferIdx]->unbindPbo();

    Refresh();
  }
  catch ( const std::exception& e )
  {
    std::stringstream ss;
    ss << "Step error: " << e.what();
    dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  }
  catch ( ... )
  {
    std::stringstream ss;
    ss << "unknown Step error: ";
    dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  }

  t.stop();
  std::stringstream ss;
  ss << "Step " << std::fixed << std::setprecision( 4 ) << t.ms() << " ms";
  dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
}

void GLCanvas::Reset()
{
  try
  {
    // map cuda resource: front pbo
    mPBOs[mFrontBufferIdx]->mapCudaResource();
    uint8_t* mappedPtr = mPBOs[mFrontBufferIdx]->getCudaMappedPointer();
    RunFillKernel( mappedPtr, 0, mTextures.front()->width(), mTextures.front()->height() );
    mPBOs[mFrontBufferIdx]->unmapCudaResource();

    mPBOs[mFrontBufferIdx]->bindPbo();
    mTextures.front()->bind();
    mTextures.front()->updateFromPBO();
    mTextures.front()->bind();
    mPBOs[mFrontBufferIdx]->unbindPbo();

    Refresh();
  }
  catch ( const std::exception& e )
  {
    std::stringstream ss;
    ss << "Reset error: " << e.what();
    dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( ss.str() );
  }
}

void GLCanvas::OnPaint( wxPaintEvent& /*event*/ )
{
  SetCurrent( *mContext );

  glClearColor( 0.05f, 0.05f, 0.05f, 1.0f );
  glClear( GL_COLOR_BUFFER_BIT );

  glBindVertexArray( mVao );
  glUseProgram( mShaderProgram );

  const math::mat4 vpMatrix = mProjectionMatrix * mViewMatrix;
  const int32_t uniformLoc( glGetUniformLocation( mShaderProgram, "vpMatrix" ) );
  glUniformMatrix4fv( uniformLoc, 1, GL_FALSE, &vpMatrix[0][0] );

  mTextures.front()->bindTextureUnit( 0 );

  glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0 );  // 6 = index count

  mTextures.front()->unbindTextureUnit();
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
  mPreviousMousePosition = math::vec2( static_cast<float>( event.GetX() ), static_cast<float>( event.GetY() ) );
  mPanningActive = true;
}

void GLCanvas::OnMouseRightUp( wxMouseEvent& /*event*/ )
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

  SetPixel( imagePos );
  Refresh();
}

void GLCanvas::OnMouseLeftUp( wxMouseEvent& /*event*/ )
{
  mDrawingActive = false;
}

void GLCanvas::OnMouseLeave( wxMouseEvent& /*event*/ )
{
  mPanningActive = mDrawingActive = false;
}
