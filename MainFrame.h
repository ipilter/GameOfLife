#pragma once

#include "WxMain.h"

class GLCanvas;

class MainFrame : public wxFrame
{
  friend class StepTimer;

public:
  MainFrame( wxWindow* parent, std::wstring title, const wxPoint& pos, const wxSize& size, const uint32_t exponent );
  virtual ~MainFrame();

  void AddLogMessage( const std::string& msg );

private:
  void OnStepTimer();
  void OnStartButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );
  void OnResetButton( wxCommandEvent& event );
  void OnColorButton( wxCommandEvent& event );
  void OnPatternComboBox( wxCommandEvent& event );
  void OnSlider( wxCommandEvent& event );

private:
  class StepTimer : public wxTimer // TODO maybe do inside GLCanvas or even deeper -> GOLEngine ?
  {
  public:
    StepTimer( MainFrame* parent );
    virtual void Notify();

  private:
    MainFrame* mParent;
  };

  GLCanvas* mGLCanvas;
  wxButton* mColorButton;
  wxComboBox* mPatternComboBox;
  wxSlider* mDeltaTimeSlider;
  wxTextCtrl* mLogTextBox;

  bool mStepTimerRuning = false;
  bool mLogStepRuntime = false;
  std::unique_ptr<StepTimer> mStepTimer;
  float mStepDeltaTime = 0.0f;
};
