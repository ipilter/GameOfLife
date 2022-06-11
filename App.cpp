#include <iostream>
#include <wx/cmdline.h>

#include "Util.h"
#include "GLCanvas.h"
#include "MainFrame.h"
#include "App.h"
#include "Logger.h"

bool App::OnInit()
{
  wxApp::OnInit();

  logger::Logger::instance() << __FUNCTION__ << "\n";

  Bind( wxEVT_KEY_DOWN, &App::OnKey, this );

  mMainFrame = new MainFrame( nullptr, L"GPU Game Of Life", wxDefaultPosition, { 1920, 1080 }, mTextureSize );
  return mMainFrame->Show( true );
}

int32_t App::OnExit()
{
  logger::Logger::instance() << __FUNCTION__ << "\n";

  mMainFrame->GetAccessible();
  return 1;
}

void App::OnKey( wxKeyEvent& event )
{
  if ( event.GetKeyCode() == WXK_ESCAPE )
  {
    Exit();
  }
}

void App::OnInitCmdLine( wxCmdLineParser& parser )
{
  static const wxCmdLineEntryDesc cmdLineDesc[] =
  {
    { wxCMD_LINE_SWITCH, "v", "verbose", "be verbose" },
    { wxCMD_LINE_SWITCH, "q", "quiet",   "be quiet" },
    { wxCMD_LINE_OPTION, "e", "exponent", "texture size exponent", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_NONE }
  };

  parser.SetDesc(cmdLineDesc);
}

bool App::OnCmdLineParsed( wxCmdLineParser& parser )
{
  wxApp::OnCmdLineParsed( parser );

  long parsedOption = 0;
  if ( parser.Found( wxT("e"), &parsedOption ) )
  {
    mTextureSize = static_cast<uint32_t>( parsedOption );
  }

  return true;
}
