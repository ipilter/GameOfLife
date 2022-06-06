#pragma once

#include <memory>
#include <fstream>

#include "Math.h"

namespace logger
{

class Logger
{
public:
  static Logger& instance()
  {
    static Logger instance; // static initialization is thread-safe in C++11. !! after VS2015 !!
    //volatile int dummy{}; // max optimization can remove this mehtod?
    return instance;
  }

  template<class T>
  Logger& operator << ( const T& t )
  {
    logStream << t;
    logStream.flush();
    return *this;
  }

  ~Logger() = default;

private:
  Logger()
  {
    logStream.open( "e:\\default.log" );
  }

  Logger( const Logger& rhs ) = delete;
  Logger( Logger&& rhs ) noexcept = delete;
  Logger& operator=( const Logger& rhs ) = delete;
  Logger& operator=( Logger&& rhs ) noexcept = delete;

  std::ofstream logStream;
};

}
