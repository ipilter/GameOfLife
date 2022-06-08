#pragma once

#include <cstdint>
#include "WxMain.h"

class MainFrame;

class App : public wxApp
{
public:
  virtual bool OnInit();

private:
  virtual int32_t OnExit();
  void OnKey( wxKeyEvent& event );

  virtual void OnInitCmdLine(wxCmdLineParser& parser);
  virtual bool OnCmdLineParsed(wxCmdLineParser& parser);

private:
  MainFrame* mMainFrame;
  uint32_t mTextureSize = 8;
  uint32_t mDeltaTime = 100;
};
