#pragma once

#include <iomanip>

#define GLM_FORCE_XYZW_ONLY
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace math
{

using ivec2 = glm::ivec2;
using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;

using mat4 = glm::mat4;
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
  for ( auto i = 0; i < 4; ++i )
  {
    if ( i != 0 )
    {
      stream << ", ";
    }
    stream << "[";
    for ( auto j = 0; j < 4; ++j )
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
