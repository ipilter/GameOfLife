#pragma once

#include "WxMain.h"

class GLCanvas;

class MainFrame : public wxFrame
{
public:
  MainFrame( wxWindow* parent, std::wstring title, const wxPoint& pos, const wxSize& size, const int exponent, const int deltaTime );

  virtual ~MainFrame();

  void AddLogMessage( const std::string& msg );

private:
  void OnStartButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );
  void OnResetButton( wxCommandEvent& event );
  void OnColorButton( wxCommandEvent& event );

private:
  class StepTimer : public wxTimer
  {
  public:
    void SetGLCanvas( GLCanvas* glCanvas );
    virtual void Notify();
  private:
    GLCanvas* mGLCanvas;
  };

  GLCanvas* mGLCanvas;
  wxButton* mColorButton;
  wxTextCtrl* mLogTextBox;
  StepTimer mStepTimer;
  int mDeltaTime;
};
