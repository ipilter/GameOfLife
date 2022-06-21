#pragma once

#include <map>
#include <vector>
#include <gl/glew.h> // must be before any OpenGL related header
#include "WxMain.h"
#include <wx/glcanvas.h>
#include <wx/overlay.h>
#include "Math.h"
#include "Pattern.h"
#include "Texture.h"
#include "PixelBufferObject.h"
#include "Mesh.h"

class GLCanvas : public wxGLCanvas
{
  class StepTimer : public wxTimer
  {
  public:
    StepTimer( GLCanvas* parent );
    virtual void Notify();

  private:
    GLCanvas* mParent;
  };

  friend class StepTimer;

public:
  GLCanvas( const uint32_t textureSize
            , wxWindow* parent
            , wxWindowID id = wxID_ANY
            , const int32_t* attribList = 0
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

  void SetPrimaryColor( const uint32_t color );
  const uint32_t GetPrimaryColor() const;
  void SetSecondaryColor( const uint32_t color );
  const uint32_t GetSecondaryColor() const;

  void SetCurrentPattern( const uint32_t idx );
  uint32_t GetPatternCount() const;
  const Pattern& GetPattern( const uint32_t idx ) const;
  void SetDrawPixelGrid( const bool drawPixelGrid );

  void Reset();
  void Random();
  void RotatePattern();
  void Start();
  void Stop();
  void SetDeltaTime( const uint32_t dt );

private:
  void InitializeGLEW();
  void CreateGeometry();
  void CreateShaderProgram();
  void CreateTextures();
  uint32_t CreateShader( uint32_t kind, const std::string& src );
  void CreatePatterns();

  math::vec2 ScreenToWorld( const math::ivec2& screenSpacePoint );
  math::ivec2 WorldToImage( const math::vec2& worldSpacePoint );

  void Step();

  void SetPixel( const math::uvec2& pixel );
  void OnPaint( wxPaintEvent& event );
  void OnSize( wxSizeEvent& event );
  void OnMouseMove( wxMouseEvent& event );
  void OnMouseRightDown( wxMouseEvent& event );
  void OnMouseRightUp( wxMouseEvent& event );
  void OnMouseMiddleDown( wxMouseEvent& event );
  void OnMouseMiddleUp( wxMouseEvent& event );
  void OnMouseLeftDown( wxMouseEvent& event );
  void OnMouseLeftUp( wxMouseEvent& event );
  void OnMouseLeave( wxMouseEvent& event );
  void OnMouseWheel( wxMouseEvent& event );
  void OnKeyDown( wxKeyEvent& event );
  void OnStepTimer();

  // opengl context
  std::unique_ptr<wxGLContext> mContext;

  // parameters
  float mQuadSize;
  uint32_t mTextureSize;
  uint32_t mTexturePatternSize;

  // mesh
  std::vector<Mesh::Ptr> mMeshes;

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
  uint32_t mPrimaryColor = 0;
  uint32_t mSecondaryColor = 0;
  uint32_t mCurrentDrawingColor = 0; // TODO index in an array instead

  // patterns
  std::vector<Pattern::Ptr> mDrawPatterns;
  Pattern mDrawPattern; // current patern used by SetPixel. Copied as it can be rotated
  bool mDrawPixelGrid = false;

  // step timer
  bool mStepTimerRuning = false;
  std::unique_ptr<StepTimer> mStepTimer;
  float mStepDeltaTime = 0.0f;
};
