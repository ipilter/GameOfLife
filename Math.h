#pragma once

#include <iomanip>
#include <random>
#include <functional>

#define GLM_FORCE_XYZW_ONLY
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace math
{
static const double Pi = 3.14159265;
static const double Pi2 = 6.28318530;

using ivec2 = glm::ivec2;
using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;

using mat4 = glm::mat4;

inline float Random()
{
  static auto uniformRandom( std::bind(  std::uniform_real_distribution<float>{ 0, 1 }, std::default_random_engine( std::random_device()() ) ) );
  return uniformRandom();
}

template<class T = float>
inline T Random( const T min, const T max )
{
  return min + static_cast<T>( std::rand() % ( max - min + 1 ) );
}

// Maps value 's' from range [a1,a2] into [b1,b2].
//      b1+(s-a1)(b2-b1)
// s'=  ----------------
//          (a2-a1)
//
template<class From, class To = From>
class RangeMapper
{
public:
  RangeMapper( const From a1, const From a2, const To b1, const To b2 )
    : m_a1( a1 )
    , m_a2( a2 )
    , m_b1( b1 )
    , m_b2( b2 )
  {}
public:
  To Map( const From s ) const
  {
    From a_norm( m_a2 - m_a1 );
    if ( a_norm == 0 )
    {
      return 0;
    }
    return To( ( m_b1 + ( s - static_cast<To>( m_a1 ) ) * ( m_b2 - m_b1 ) ) / static_cast<To>( a_norm ) );
  }
private:
  From m_a1;
  From m_a2;
  To m_b1;
  To m_b2;
};

}

inline std::ostream& operator << ( std::ostream& stream, const math::ivec2& v )
{
  stream << "[" << v.x << ", " << v.y << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::uvec2& v )
{
  stream << "[" << v.x << ", " << v.y << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::vec2& v )
{
  stream << std::fixed << std::setprecision(6) << "[" << v.x << ", " << v.y << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::vec3& v )
{
  stream << std::fixed << std::setprecision(6) << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::vec4& v )
{
  stream << std::fixed << std::setprecision(6) << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::mat4& m )
{
  stream << std::fixed << std::setprecision(6) << "[";
  for ( uint32_t i = 0; i < 4; ++i )
  {
    if ( i != 0 )
    {
      stream << ", ";
    }
    stream << "[";
    for ( uint32_t j = 0; j < 4; ++j )
    {
      if ( j != 0 )
      {
        stream << ", ";
      }
      stream << m[i][j];
    }
    stream << "]";
  }
  stream << "]";
  return stream;
}
