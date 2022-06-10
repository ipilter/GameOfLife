#pragma once

#include "WxMain.h"

class GLCanvas;

class MainFrame : public wxFrame
{
public:
  MainFrame( wxWindow* parent, std::wstring title, const wxPoint& pos, const wxSize& size, const uint32_t exponent, const uint32_t deltaTime );
  virtual ~MainFrame();

  void AddLogMessage( const std::string& msg );

private:
  void OnStartButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );
  void OnResetButton( wxCommandEvent& event );
  void OnColorButton( wxCommandEvent& event );
  void OnPatternComboBox( wxCommandEvent& event );

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
  wxComboBox* mPatternComboBox;
  wxTextCtrl* mLogTextBox;
  StepTimer mStepTimer;
  uint32_t mDeltaTime;
};
