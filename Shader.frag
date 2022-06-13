#version 450 core

uniform sampler2D textureData;
uniform sampler2D checkerboardData;
uniform int isCheckerboard;

in vec2 uv;

out vec4 fragColor;

void main()
{
  if ( isCheckerboard > 0 )
  {
    fragColor = max ( texture ( textureData, uv ), texture( checkerboardData, uv ) );
  }
  else
  {
    fragColor = texture ( textureData, uv );
  }
}
