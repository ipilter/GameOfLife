#include <sstream>
#include "MainFrame.h"
#include <wx/colordlg.h>
#include <wx/combobox.h>
#include "GLCanvas.h"
#include "Logger.h"
#include "Timer.h"
#include "Util.h"

MainFrame::MainFrame( wxWindow* parent, std::wstring title, const wxPoint& pos, const wxSize& size, const uint32_t textureSize )
  : wxFrame( parent, wxID_ANY, title, pos, size )
  , mGLCanvas( nullptr )
  , mPrimaryColorButton( nullptr )
  , mSecondaryColorButton( nullptr )
  , mLogTextBox( nullptr )
{
  try
  {
    logger::Logger::Instance() << __FUNCTION__ << "\n";

    auto* mainPanel = new wxPanel( this, wxID_ANY );
    mLogTextBox = new wxTextCtrl( mainPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE );
    mGLCanvas = new GLCanvas( textureSize, mainPanel, wxID_ANY );

    auto* mainSizer = new wxBoxSizer( wxVERTICAL );
    mainSizer->Add( mGLCanvas, 90, wxEXPAND );
    mainSizer->Add( mLogTextBox, 10, wxEXPAND );
    mainPanel->SetSizer( mainSizer );

    auto* controlPanel = new wxPanel( this, wxID_ANY );
    auto* resetBtn = new wxButton( controlPanel, wxID_ANY, "Reset" );
    auto* startBtn = new wxButton( controlPanel, wxID_ANY, "Start" );
    auto* stopBtn = new wxButton( controlPanel, wxID_ANY, "Stop" );
    auto* randomBtn = new wxButton( controlPanel, wxID_ANY, "Random" );
    mPrimaryColorButton = new wxButton( controlPanel, wxID_ANY );
    mPrimaryColorButton->SetBackgroundColour( wxColor( util::Component(mGLCanvas->GetPrimaryColor(), 0)
                                                       , util::Component(mGLCanvas->GetPrimaryColor(), 1)
                                                       , util::Component(mGLCanvas->GetPrimaryColor(), 2) ) );
    
    mSecondaryColorButton = new wxButton( controlPanel, wxID_ANY );
    mSecondaryColorButton->SetBackgroundColour( wxColor( util::Component(mGLCanvas->GetSecondaryColor(), 0)
                                                         , util::Component(mGLCanvas->GetSecondaryColor(), 1)
                                                         , util::Component(mGLCanvas->GetSecondaryColor(), 2) ) );

    mPatternComboBox = new wxComboBox( controlPanel, wxID_ANY );
    mPatternComboBox->Bind( wxEVT_COMBOBOX_CLOSEUP, &MainFrame::OnPatternComboBox, this );
    
    mDeltaTimeSlider = new wxSlider( controlPanel, wxID_ANY, 10, 1, 100 );
    mDeltaTimeSlider->Bind( wxEVT_SLIDER, &MainFrame::OnSlider, this );

    mPixelGridCheckBox = new wxCheckBox( controlPanel, wxID_ANY, "Pixel Grid" );

    resetBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnResetButton, this );
    startBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStartButton, this );
    stopBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this );
    randomBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnRandomButton, this );
    mPrimaryColorButton->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnPrimaryColorButton, this );
    mSecondaryColorButton->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnSecondaryColorButton, this );

    auto percentage = mDeltaTimeSlider->GetValue() / 100.0f;
    mGLCanvas->SetDeltaTime( ( 500u - 0u ) * percentage ); // TODO fix

    mPixelGridCheckBox->Bind( wxEVT_COMMAND_CHECKBOX_CLICKED, &MainFrame::OnPixelCheckBox, this ); 
    mPixelGridCheckBox->SetForegroundColour( wxColor( 255, 255, 255 ) );
    mPixelGridCheckBox->SetValue( true );
    mGLCanvas->SetDrawPixelGrid( mPixelGridCheckBox->IsChecked() );

    auto* controlSizer = new wxBoxSizer( wxVERTICAL ); // TODO List of controls instead these
    controlSizer->Add( resetBtn, 0, wxEXPAND );
    controlSizer->Add( startBtn, 0, wxEXPAND );
    controlSizer->Add( stopBtn, 0, wxEXPAND );
    controlSizer->Add( randomBtn, 0, wxEXPAND );
    controlSizer->Add( mPrimaryColorButton, 0, wxEXPAND );
    controlSizer->Add( mSecondaryColorButton, 0, wxEXPAND );
    controlSizer->Add( mPatternComboBox, 0, wxEXPAND );
    controlSizer->Add( mDeltaTimeSlider, 0, wxEXPAND );
    controlSizer->Add( mPixelGridCheckBox, 0, wxEXPAND );
    controlPanel->SetSizer( controlSizer );

    auto* sizer = new wxBoxSizer( wxHORIZONTAL );
    sizer->Add( mainPanel, 1, wxEXPAND );
    sizer->Add( controlPanel, 0, wxEXPAND );

    this->SetSizer( sizer );

    controlPanel->SetBackgroundColour( wxColor( 21, 21, 21 ) );
    mLogTextBox->SetBackgroundColour( wxColor( 21, 21, 21 ) );
    mLogTextBox->SetForegroundColour( wxColor( 180, 180, 180 ) );

    for ( uint32_t idx = 0; idx < mGLCanvas->GetPatternCount(); ++idx )
    {
      mPatternComboBox->Append( mGLCanvas->GetPattern( idx ).GetName() );
    }
    mPatternComboBox->SetSelection( 0 );
    mGLCanvas->SetCurrentPattern( 0 );
  }
  catch ( const std::exception& e )
  {
    logger::Logger::Instance() << "MainFrame construction error: " << e.what() << "\n";
  }
  
  std::stringstream ss;
  ss << "GLCanvas window size: " << math::vec2( GetSize().GetWidth(), GetSize().GetHeight() );
  AddLogMessage( ss.str() );

  logger::Logger::Instance() << ss.str() << "\n";
}

MainFrame::~MainFrame()
{
  logger::Logger::Instance() << __FUNCTION__ << "\n";
}

void MainFrame::AddLogMessage( const std::string& msg )
{
  mLogTextBox->WriteText( ( msg + "\n" ) );
  logger::Logger::Instance() << msg << "\n";
}

void MainFrame::OnStartButton( wxCommandEvent& /*event*/ )
{
  mGLCanvas->Start();
}

void MainFrame::OnStopButton(wxCommandEvent& /*event*/)
{
  mGLCanvas->Stop();
}

void MainFrame::OnResetButton(wxCommandEvent& /*event*/)
{
  try
  {
    mGLCanvas->Reset();
  }
  catch ( const std::exception& e )
  {
    std::stringstream ss;
    ss << "Reset error: " << e.what();
    AddLogMessage( ss.str() );
  }
  AddLogMessage( "simulation reseted" );
}

void MainFrame::OnRandomButton( wxCommandEvent& event )
{
  try
  {
    mGLCanvas->Random();
  }
  catch ( const std::exception& e )
  {
    std::stringstream ss;
    ss << "Random error: " << e.what();
    AddLogMessage( ss.str() );
  }
}

void MainFrame::OnPrimaryColorButton( wxCommandEvent& event )
{
  const uint32_t pc = mGLCanvas->GetPrimaryColor();
  const wxColour wxc( wxGetColourFromUser( this, wxColor( util::Component( pc, 0 ), util::Component( pc, 1 ), util::Component( pc, 2 ) ) ) );
  mPrimaryColorButton->SetBackgroundColour( wxc );
  mGLCanvas->SetPrimaryColor( util::Color( wxc.Red(), wxc.Green(), wxc.Blue() ) );
}

void MainFrame::OnSecondaryColorButton( wxCommandEvent& event )
{
  const uint32_t sc = mGLCanvas->GetSecondaryColor();
  const wxColour wxc( wxGetColourFromUser( this, wxColor( util::Component( sc, 0 ), util::Component( sc, 1 ), util::Component( sc, 2 ) ) ) );
  mSecondaryColorButton->SetBackgroundColour( wxc );
  mGLCanvas->SetSecondaryColor( util::Color( wxc.Red(), wxc.Green(), wxc.Blue() ) );
}

void MainFrame::OnPatternComboBox( wxCommandEvent& /*event*/ )
{
  mGLCanvas->SetCurrentPattern( mPatternComboBox->GetSelection() ); // decopule index and content
}

void MainFrame::OnSlider( wxCommandEvent& event )
{
  // TODO redundant and not correct
  auto percentage = mDeltaTimeSlider->GetValue() / 100.0f;
  mGLCanvas->SetDeltaTime( ( 500u - 0u ) * percentage );
}

void MainFrame::OnPixelCheckBox( wxCommandEvent& event )
{
  mGLCanvas->SetDrawPixelGrid( mPixelGridCheckBox->IsChecked() );
}
