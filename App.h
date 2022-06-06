#pragma once

#include "WxMain.h"

class MainFrame;

class App : public wxApp
{
public:
  virtual bool OnInit();

private:
  virtual int OnExit();
  void OnKey( wxKeyEvent& event );

  virtual void OnInitCmdLine(wxCmdLineParser& parser);
  virtual bool OnCmdLineParsed(wxCmdLineParser& parser);

private:
  MainFrame* mMainFrame;
  int mTextureSize;
  int mDeltaTime;
};
