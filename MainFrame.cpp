#include <sstream>
#include "MainFrame.h"
#include <wx/colordlg.h>
#include <wx/combobox.h>
#include "GLCanvas.h"
#include "Logger.h"
#include "Timer.h"
#include "Util.h"

MainFrame::MainFrame( wxWindow* parent, std::wstring title, const wxPoint& pos, const wxSize& size, const uint32_t exponent )
  : wxFrame( parent, wxID_ANY, title, pos, size )
  , mGLCanvas( nullptr )
  , mColorButton( nullptr )
  , mLogTextBox( nullptr )
{
  try
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
    
    mDeltaTimeSlider = new wxSlider( controlPanel, wxID_ANY, 10, 1, 100 );
    mDeltaTimeSlider->Bind( wxEVT_SLIDER, &MainFrame::OnSlider, this );

    auto percentage = mDeltaTimeSlider->GetValue() / 100.0f;
    mStepDeltaTime = (500u - 1u) * percentage; // percentage of min max milliseconds

    resetBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnResetButton, this );
    startBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStartButton, this );
    stopBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this );
    mColorButton->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnColorButton, this );

    auto* controlSizer = new wxBoxSizer( wxVERTICAL ); // TODO List of controls instead these
    controlSizer->Add( resetBtn, 0, wxEXPAND );
    controlSizer->Add( startBtn, 0, wxEXPAND );
    controlSizer->Add( stopBtn, 0, wxEXPAND );
    controlSizer->Add( mColorButton, 0, wxEXPAND );
    controlSizer->Add( mPatternComboBox, 0, wxEXPAND );
    controlSizer->Add( mDeltaTimeSlider, 0, wxEXPAND );
    controlPanel->SetSizer( controlSizer );

    auto* sizer = new wxBoxSizer( wxHORIZONTAL );
    sizer->Add( mainPanel, 1, wxEXPAND );
    sizer->Add( controlPanel, 0, wxEXPAND );

    this->SetSizer( sizer );

    mStepTimer = std::make_unique<StepTimer>( this );

    controlPanel->SetBackgroundColour( wxColor( 21, 21, 21 ) );

    mLogTextBox->SetBackgroundColour( wxColor( 21, 21, 21 ) );
    mLogTextBox->SetForegroundColour( wxColor( 180, 180, 180 ) );

    for ( uint32_t idx = 0; idx < mGLCanvas->GetPatternCount(); ++idx )
    {
      mPatternComboBox->Append( mGLCanvas->GetPattern( idx ).name() );
    }
    mPatternComboBox->SetSelection( 0 );
    mGLCanvas->SetCurrentPattern( 0 );
  }
  catch ( const std::exception& e )
  {
    logger::Logger::instance() << "MainFrame construction error: " << e.what() << "\n";
  }
  
  std::stringstream ss;
  ss << "GLCanvas window size: " << math::vec2( GetSize().GetWidth(), GetSize().GetHeight() );
  AddLogMessage( ss.str() );

  logger::Logger::instance() << ss.str() << "\n";
}

MainFrame::~MainFrame()
{
  logger::Logger::instance() << __FUNCTION__ << "\n";
}

void MainFrame::AddLogMessage( const std::string& msg )
{
  mLogTextBox->WriteText( ( msg + "\n" ) );
}

void MainFrame::OnStepTimer()
{
  Timer* t;
  if ( mLogStepRuntime )
  {
    if ( t = nullptr )
    {
      t = new Timer();
    }
    t->start();
  }
  try
  {
    mGLCanvas->Step();
  }
  catch ( const std::exception& e )
  {
    mStepTimer->Stop();

    std::stringstream ss;
    ss << "Step error: " << e.what();
    AddLogMessage( ss.str() );
  }
  catch ( ... )
  {
    mStepTimer->Stop();

    std::stringstream ss;
    ss << "unknown Step error: ";
    AddLogMessage( ss.str() );
  }

  if ( mLogStepRuntime )
  {
    t->stop();
    std::stringstream ss;
    ss << "Step " << std::fixed << std::setprecision( 4 ) << t->ms() << " ms";
    AddLogMessage( ss.str() );
  }
}

void MainFrame::OnStartButton( wxCommandEvent& /*event*/ )
{
  if ( mStepTimerRuning )
  {
    return;
  }

  mStepTimer->Start( mStepDeltaTime );
  mStepTimerRuning = true;
  AddLogMessage( std::string("simulation started ") + util::ToString<float>( mStepDeltaTime ) + " ms");
}

void MainFrame::OnStopButton(wxCommandEvent& /*event*/)
{
  if ( !mStepTimerRuning )
  {
    return;
  }

  mStepTimer->Stop();
  mStepTimerRuning = false;

  AddLogMessage( "simulation stopped" );
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

void MainFrame::OnSlider( wxCommandEvent& event )
{
  // TODO redundant
  auto percentage = mDeltaTimeSlider->GetValue() / 100.0f;
  mStepDeltaTime = (500u - 0u) * percentage; // percentage of min max milliseconds
  if ( mStepTimerRuning )
  {
    OnStopButton( event );
    OnStartButton( event );
  }

  std::stringstream ss;
  ss << "slider: " << mStepDeltaTime;
  AddLogMessage( ss.str() );
}

// StepTimer
MainFrame::StepTimer::StepTimer( MainFrame* parent )
  : mParent( parent )
{}

void MainFrame::StepTimer::Notify()
{
  mParent->OnStepTimer();
}
