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
    logger::Logger::instance() << __FUNCTION__ << "\n";

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
    mPrimaryColorButton->SetBackgroundColour( wxColor( mGLCanvas->GetPrimaryColor().x, mGLCanvas->GetPrimaryColor().y, mGLCanvas->GetPrimaryColor().z ) );
    
    mSecondaryColorButton = new wxButton( controlPanel, wxID_ANY );
    mSecondaryColorButton->SetBackgroundColour( wxColor( mGLCanvas->GetSecondaryColor().x, mGLCanvas->GetSecondaryColor().y, mGLCanvas->GetSecondaryColor().z ) );

    mPatternComboBox = new wxComboBox( controlPanel, wxID_ANY );
    mPatternComboBox->Bind( wxEVT_COMBOBOX_CLOSEUP, &MainFrame::OnPatternComboBox, this );
    
    mDeltaTimeSlider = new wxSlider( controlPanel, wxID_ANY, 10, 1, 100 );
    mDeltaTimeSlider->Bind( wxEVT_SLIDER, &MainFrame::OnSlider, this );

    mPixelGridCheckBox = new wxCheckBox( controlPanel, wxID_ANY, "Pixel grid" );

    resetBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnResetButton, this );
    startBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStartButton, this );
    stopBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this );
    randomBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnRandomButton, this );
    mPrimaryColorButton->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnPrimaryColorButton, this );
    mSecondaryColorButton->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnSecondaryColorButton, this );

    auto percentage = mDeltaTimeSlider->GetValue() / 100.0f;
    mStepDeltaTime = (500u - 0u) * percentage; // percentage of min max milliseconds

    mPixelGridCheckBox->Bind( wxEVT_COMMAND_CHECKBOX_CLICKED, &MainFrame::OnPixelCheckBox, this ); 
    mPixelGridCheckBox->SetForegroundColour( wxColor( 255, 255, 255 ) );

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
  const auto color( wxGetColourFromUser( this, wxColor( mGLCanvas->GetPrimaryColor().x, mGLCanvas->GetPrimaryColor().y, mGLCanvas->GetPrimaryColor().z ) ) );
  mPrimaryColorButton->SetBackgroundColour( color );

  mGLCanvas->SetPrimaryColor( math::uvec3( color.Red(), color.Green(), color.Blue() ) );
}

void MainFrame::OnSecondaryColorButton( wxCommandEvent& event )
{
  const auto color( wxGetColourFromUser( this, wxColor( mGLCanvas->GetSecondaryColor().x, mGLCanvas->GetSecondaryColor().y, mGLCanvas->GetSecondaryColor().z ) ) );
  mSecondaryColorButton->SetBackgroundColour( color );

  mGLCanvas->SetSecondaryColor( math::uvec3( color.Red(), color.Green(), color.Blue() ) );
}

void MainFrame::OnPatternComboBox( wxCommandEvent& /*event*/ )
{
  mGLCanvas->SetCurrentPattern( mPatternComboBox->GetSelection() ); // decopule index and content
}

void MainFrame::OnSlider( wxCommandEvent& event )
{
  // TODO redundant and not correct
  auto percentage = mDeltaTimeSlider->GetValue() / 100.0f;
  mStepDeltaTime = (500u - 0u) * percentage; // percentage of min max milliseconds
  if ( mStepTimerRuning )
  {
    OnStopButton( event );
    OnStartButton( event );
  }
}

void MainFrame::OnPixelCheckBox( wxCommandEvent& event )
{
  mGLCanvas->SetDrawPixelGrid( mPixelGridCheckBox->IsChecked() );
}

// StepTimer
MainFrame::StepTimer::StepTimer( MainFrame* parent )
  : mParent( parent )
{}

void MainFrame::StepTimer::Notify()
{
  mParent->OnStepTimer();
}
