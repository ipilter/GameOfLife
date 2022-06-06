
#include "timer.h"

#include <windows.h>

namespace detail
{
class TimerImpl
{
public:
  TimerImpl();

  void start();
  void stop();

public:
  double us( void );

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

void TimerImpl::start()
{
  m_stopped = false;
  QueryPerformanceCounter( &m_begin );
}

void TimerImpl::stop()
{
  QueryPerformanceCounter( &m_end );
  m_stopped = true;
}

double TimerImpl::us( void )
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

void Timer::Timer::start()
{
  m_pImpl->start();
}

void Timer::Timer::stop()
{
  m_pImpl->stop();
}

double Timer::us( void )
{
  return m_pImpl->us();
}

double Timer::ms( void )
{
  return m_pImpl->us() * 0.001;
}

double Timer::s( void )
{
  return m_pImpl->us() * 0.000001;
}