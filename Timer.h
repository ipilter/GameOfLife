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
  void Start();
  void Stop();

public:
  double Us( void );
  double Ms( void );
  double S( void );

private:
  detail::TimerImpl* m_pImpl;
};
