#pragma once

#include "WxMain.h"

class GLCanvas;

class MainFrame : public wxFrame
{
  friend class StepTimer;

public:
  MainFrame( wxWindow* parent, std::wstring title, const wxPoint& pos, const wxSize& size, const uint32_t textureSize );
  virtual ~MainFrame();

  void AddLogMessage( const std::string& msg );

private:
  void OnStepTimer();
  void OnStartButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );
  void OnResetButton( wxCommandEvent& event );
  void OnRandomButton( wxCommandEvent& event );
  void OnPrimaryColorButton( wxCommandEvent& event );
  void OnSecondaryColorButton( wxCommandEvent& event );
  void OnPatternComboBox( wxCommandEvent& event );
  void OnSlider( wxCommandEvent& event );
  void OnPixelCheckBox( wxCommandEvent& event );

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
  wxButton* mPrimaryColorButton;
  wxButton* mSecondaryColorButton;
  wxComboBox* mPatternComboBox;
  wxSlider* mDeltaTimeSlider;
  wxTextCtrl* mLogTextBox;
  wxCheckBox* mPixelGridCheckBox;

  bool mStepTimerRuning = false;
  std::unique_ptr<StepTimer> mStepTimer;
  float mStepDeltaTime = 0.0f;
};
