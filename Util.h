#pragma once

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace util
{
namespace detail
{
constexpr std::uint32_t redBits  { 0x000000FF };
constexpr std::uint32_t greenBits{ 0x0000FF00 };
constexpr std::uint32_t blueBits { 0x00FF0000 };
constexpr std::uint32_t alphaBits{ 0xFF000000 };

// R: 0x000000FF
// G: 0x0000FF00
// B: 0x00FF0000
// A: 0xFF000000
union U
{
#ifdef _DEBUG
  // TODO endiannes
  const static uint32_t i = 1;
  static_assert( ( i & 0x1 ) == 1 );
#endif

public:
  explicit U( const uint32_t i )
    : mInt( i )
  {}
  explicit U( const uint8_t r, const uint8_t g, const uint8_t b, const uint8_t a )
  {
    mChr[0] = r;
    mChr[1] = g;
    mChr[2] = b;
    mChr[3] = a;
  }
  uint32_t Int() const
  {
    return mInt;
  }
  uint32_t Chr( const uint32_t& idx ) const
  {
    return mChr[idx];
  }
private:
  uint32_t mInt;
  uint8_t mChr[4];
};

} // ::detail

inline uint8_t Component( const uint32_t& color, const uint32_t& idx )
{
  const detail::U u( color );
  return u.Chr( idx ); // TODO alert system
}

inline uint32_t Color( const uint8_t r = 0, const uint8_t g = 0, const uint8_t b = 0, const uint8_t a = 255 )
{
  const detail::U u( r, g, b, a );
  return u.Int();
}

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
  const size_t count( str.length() );
  Write_t( stream, count );
  stream.write( str.c_str(), count );
}

template<>
inline void Write_t( std::ofstream& stream, const std::vector<bool>& array )
{
  const size_t count( array.size() );
  Write_t( stream, count );
  for ( size_t i( 0 ); i < count; ++i )
  {
    Write_t( stream, array[i] );
  }
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
  Read_t( stream, dummy ); // TODO why is this. I`m sure it is needed but not remember why. /0 at the end of the cstr? Comment it !!
}

template<>
inline void Read_t( std::ifstream& stream, std::vector<bool>& array )
{
  size_t count( 0 );
  Read_t( stream, count );
  array.resize( count, 0 );

  for ( size_t i( 0 ); i < count; ++i )
  {
    bool v = false;
    Read_t( stream, v );
    array[i] = v;
  }
}

inline float RoundToNearestMultiple( const float val, const float n )
{
  return std::floor( val / n ) * n;
}

}
