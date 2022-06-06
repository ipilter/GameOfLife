#pragma once

#include <sstream>
#include <fstream>
#include <stdexcept>

namespace util
{

template<class T>
inline T FromString( const std::string& str )
{
  T t( 0 );
  if ( !( std::istringstream( str ) >> t ) )
  {
    throw std::runtime_error( std::string( "cannot parse " ) + str + " as number" );
  }
  return t;
}

template<class T>
inline std::string ToString( const T& t )
{
  std::stringstream ss;
  ss << t;
  return ss.str();
}

template<class T>
inline T Random( const T min, const T max )
{
  if ( max < min )
  {
    std::stringstream ss;
    ss << "random range. min = " << min << "  max = " << max;
    throw std::runtime_error( ss.str() );
  }

  static bool Initialized = false;
  if ( !Initialized )
  {
    srand( static_cast<unsigned>( std::time( nullptr ) ) );
    Initialized = true;
  }
  return min + T( std::rand() ) / T( RAND_MAX ) * ( max - min );
}

inline std::string ReadTextFile( const std::string& path )
{
  std::ifstream is( path );
  if ( !is.is_open() )
  {
    throw std::runtime_error( std::string( "cannot open file: " ) + path );
  }
  return std::string( ( std::istreambuf_iterator<char>( is ) ), std::istreambuf_iterator<char>() );
}

template<class T>
inline T Clamp( const T a, const T b, const T v )
{
  return v < a ? a : v > b ? b : v;
}

template<typename T>
inline void Write_t( std::ofstream& stream, const T& t )
{
  stream.write( reinterpret_cast<const char*>( &t ), sizeof( T ) );
}

template<>
inline void Write_t( std::ofstream& stream, const std::string& str )
{
  size_t count( str.length() );
  Write_t( stream, count );
  stream.write( str.c_str(), count );
}

template<typename T>
inline void Read_t( std::ifstream& stream, T& t )
{
  stream.read( reinterpret_cast<char*> ( &t ), sizeof( T ) );
}

template<>
inline void Read_t( std::ifstream& stream, std::string& str )
{
  size_t count( 0 );
  Read_t( stream, count );
  str.reserve( count );

  std::istreambuf_iterator<char> chars( stream );
  std::copy_n( chars, count, std::back_inserter<std::string>( str ) );

  char dummy( 0 );
  Read_t( stream, dummy );
}

}