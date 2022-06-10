#include <sstream>
#include "MainFrame.h"
#include <wx/colordlg.h>
#include <wx/combobox.h>
#include "GLCanvas.h"
#include "Logger.h"

MainFrame::MainFrame( wxWindow* parent, std::wstring title, const wxPoint& pos, const wxSize& size, const uint32_t exponent, const uint32_t deltaTime )
  : wxFrame( parent, wxID_ANY, title, pos, size )
  , mGLCanvas( nullptr )
  , mColorButton( nullptr )
  , mLogTextBox( nullptr )
  , mDeltaTime( deltaTime )
{
  logger::Logger::instance() << __FUNCTION__ << "\n";

  auto* mainPanel = new wxPanel( this, wxID_ANY );
  mLogTextBox = new wxTextCtrl( mainPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE );
  mGLCanvas = new GLCanvas( exponent, mainPanel, wxID_ANY );

  auto* mainSizer = new wxBoxSizer( wxVERTICAL );
  mainSizer->Add( mGLCanvas, 90, wxEXPAND );
  mainSizer->Add( mLogTextBox, 10, wxEXPAND );
  mainPanel->SetSizer( mainSizer );

  auto* controlPanel = new wxPanel( this, wxID_ANY );
  auto* resetBtn = new wxButton( controlPanel, wxID_ANY, "Reset" );
  auto* startBtn = new wxButton( controlPanel, wxID_ANY, "Start" );
  auto* stopBtn = new wxButton( controlPanel, wxID_ANY, "Stop" );
  mColorButton = new wxButton( controlPanel, wxID_ANY );
  mColorButton->SetBackgroundColour( wxColor( mGLCanvas->GetDrawColor().x, mGLCanvas->GetDrawColor().y, mGLCanvas->GetDrawColor().z ) );
  mPatternComboBox = new wxComboBox( controlPanel, wxID_ANY );
  mPatternComboBox->Bind( wxEVT_COMBOBOX_CLOSEUP, &MainFrame::OnPatternComboBox, this );

  resetBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnResetButton, this );
  startBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStartButton, this );
  stopBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this );
  mColorButton->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnColorButton, this );

  auto* controlSizer = new wxBoxSizer( wxVERTICAL );
  controlSizer->Add( resetBtn, 0, wxEXPAND );
  controlSizer->Add( startBtn, 0, wxEXPAND );
  controlSizer->Add( stopBtn, 0, wxEXPAND );
  controlSizer->Add( mColorButton, 0, wxEXPAND );
  controlSizer->Add( mPatternComboBox, 0, wxEXPAND );
  controlPanel->SetSizer( controlSizer );

  auto* sizer = new wxBoxSizer( wxHORIZONTAL );
  sizer->Add( mainPanel, 1, wxEXPAND );
  sizer->Add( controlPanel, 0, wxEXPAND );

  this->SetSizer( sizer );

  mStepTimer.SetGLCanvas( mGLCanvas );

  controlPanel->SetBackgroundColour( wxColor( 21, 21, 21 ) );

  mLogTextBox->SetBackgroundColour( wxColor( 21, 21, 21 ) );
  mLogTextBox->SetForegroundColour( wxColor( 180, 180, 180 ) );

  for ( uint32_t idx = 0; idx < mGLCanvas->GetPatternCount(); ++idx )
  {
    mPatternComboBox->Append(  mGLCanvas->GetPattern( idx ).name() );
  }
  mPatternComboBox->SetSelection( 0 );
}

MainFrame::~MainFrame()
{
  logger::Logger::instance() << __FUNCTION__ << "\n";
}

void MainFrame::AddLogMessage( const std::string& msg )
{
  mLogTextBox->WriteText( ( msg + "\n" ) );
}

void MainFrame::OnStartButton( wxCommandEvent& /*event*/ )
{
  mStepTimer.Start( mDeltaTime );
}

void MainFrame::OnStopButton(wxCommandEvent& /*event*/)
{
  mStepTimer.Stop();
}

void MainFrame::OnResetButton(wxCommandEvent& /*event*/)
{
  mGLCanvas->Reset();
}

void MainFrame::OnColorButton( wxCommandEvent& event )
{
  const auto color( wxGetColourFromUser( this, wxColor( mGLCanvas->GetDrawColor().x, mGLCanvas->GetDrawColor().y, mGLCanvas->GetDrawColor().z ) ) );
  mColorButton->SetBackgroundColour( color );

  mGLCanvas->SetDrawColor( math::uvec3( color.Red(), color.Green(), color.Blue() ) );
}

void MainFrame::OnPatternComboBox( wxCommandEvent& /*event*/ )
{
  mGLCanvas->SetCurrentPattern( mPatternComboBox->GetSelection() ); // decopule index and content
}

// StepTimer //
void MainFrame::StepTimer::Notify()
{
  mGLCanvas->Step();
}

void MainFrame::StepTimer::SetGLCanvas( GLCanvas* glCanvas )
{
  mGLCanvas = glCanvas;
}
