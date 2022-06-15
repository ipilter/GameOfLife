#pragma once

#include "WxMain.h"

class GLCanvas;

class MainFrame : public wxFrame
{
public:
  MainFrame( wxWindow* parent, std::wstring title, const wxPoint& pos, const wxSize& size, const uint32_t textureSize );
  virtual ~MainFrame();

  void AddLogMessage( const std::string& msg );

private:
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
  GLCanvas* mGLCanvas;
  wxButton* mPrimaryColorButton;
  wxButton* mSecondaryColorButton;
  wxComboBox* mPatternComboBox;
  wxSlider* mDeltaTimeSlider;
  wxTextCtrl* mLogTextBox;
  wxCheckBox* mPixelGridCheckBox;
};
