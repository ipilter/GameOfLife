#pragma once

namespace detail
{
class TimerImpl;
}

class Timer
{
public:
  Timer();

public:
  void start();
  void stop();

public:
  double us( void );
  double ms( void );
  double s( void );

private:
  detail::TimerImpl* m_pImpl;
};
