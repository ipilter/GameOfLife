
#include "timer.h"

#include <windows.h>

namespace detail
{
class TimerImpl
{
public:
  TimerImpl();

  void Start();
  void Stop();

public:
  double Us( void );

private:
  double m_start_time;
  double m_end_time;
  bool m_stopped;

  LARGE_INTEGER m_frequency;
  LARGE_INTEGER m_begin;
  LARGE_INTEGER m_end;
};

TimerImpl::TimerImpl()
  : m_start_time( 0 )
  , m_stopped( true )
  , m_end_time( 0 )
  , m_frequency()
  , m_begin()
  , m_end()
{
  QueryPerformanceFrequency( &m_frequency );
  m_begin.QuadPart = 0;
  m_end.QuadPart = 0;
}

void TimerImpl::Start()
{
  m_stopped = false;
  QueryPerformanceCounter( &m_begin );
}

void TimerImpl::Stop()
{
  QueryPerformanceCounter( &m_end );
  m_stopped = true;
}

double TimerImpl::Us( void )
{
  if ( !m_stopped )
    QueryPerformanceCounter( &m_end );

  m_start_time = m_begin.QuadPart * ( 1000000.0 / m_frequency.QuadPart );
  m_end_time = m_end.QuadPart * ( 1000000.0 / m_frequency.QuadPart );

  return m_end_time - m_start_time;
}
} // ::detail

Timer::Timer()
  : m_pImpl( new detail::TimerImpl() )
{}

void Timer::Timer::Start()
{
  m_pImpl->Start();
}

void Timer::Timer::Stop()
{
  m_pImpl->Stop();
}

double Timer::Us( void )
{
  return m_pImpl->Us();
}

double Timer::Ms( void )
{
  return m_pImpl->Us() * 0.001;
}

double Timer::S( void )
{
  return m_pImpl->Us() * 0.000001;
}
