#pragma once

#include <map>
#include <vector>
#include <gl/glew.h> // must be before any OpenGL related header
#include "WxMain.h"
#include <wx/glcanvas.h>

#include "Math.h"
#include "Pattern.h"
#include "Texture.h"
#include "PixelBufferObject.h"

class GLCanvas : public wxGLCanvas
{
public:
  GLCanvas( const uint32_t textureExponent
            , wxWindow* parent
            , wxWindowID id = wxID_ANY
            , const int* attribList = 0
            , const wxPoint& pos = wxDefaultPosition
            , const wxSize& size = wxDefaultSize
            , long style = 0L
            , const wxString& name = L"GLCanvas"
            , const wxPalette& palette = wxNullPalette );

  virtual ~GLCanvas();

  GLCanvas( const GLCanvas& ) = delete;
  GLCanvas( GLCanvas&& ) = delete;
  GLCanvas& operator = ( const GLCanvas& ) = delete;
  GLCanvas& operator = ( GLCanvas&& ) = delete;

  void SetDrawColor( const math::uvec3& color );
  const math::uvec3& GetDrawColor() const;
  void SetCurrentPattern( const uint32_t idx );
  uint32_t GetPatternCount() const;
  const Pattern& GetPattern( const uint32_t idx ) const;

  void Step();
  void Reset();

private:
  void InitializeGLEW();
  void CreateGeometry();
  void CreateShaderProgram();
  void CreateTexture();
  uint32_t CreateShader( uint32_t kind, const std::string& src );

  math::vec2 ScreenToWorld( const math::ivec2& screenSpacePoint );
  math::ivec2 WorldToImage( const math::vec2& worldSpacePoint );

  void SetPixel( const math::uvec2& pixel );
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
  float mQuadSize = 1.0f;
  uint32_t mTextureExponent = 1;

  // mesh
  uint32_t mVbo = 0;
  uint32_t mIbo = 0;
  uint32_t mVao = 0;

  // textures
  std::vector<Texture::Ptr> mTextures;

  // pixel buffers
  std::vector<PixelBufferObject::Ptr> mPBOs;
  uint32_t mFrontBufferIdx = 0;
  uint32_t mBackBufferIdx = 0;

  // shader
  uint32_t mVertexShader = 0;
  uint32_t mFragmentxShader = 0;
  uint32_t mShaderProgram = 0;

  // view
  math::mat4 mProjectionMatrix = math::mat4( 1.0f );
  math::mat4 mViewMatrix = math::mat4( 1.0f );

  // control
  bool mPanningActive = false;
  math::vec2 mPreviousMousePosition = math::vec2( 0.0 );
  bool mDrawingActive = false;
  math::uvec3 mDrawColor = math::uvec3( 255, 255, 255 );

  // patterns
  std::vector<Pattern::Ptr> mDrawPatterns;
  size_t mDrawPatternIdx = 0;
};
