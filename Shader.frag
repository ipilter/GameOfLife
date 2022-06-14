#version 450 core

uniform sampler2D textureData;
uniform sampler2D checkerboardData;
uniform int isCheckerboard;

in vec2 fWorldUV;
in vec2 fPixelPatternUV;

out vec4 fragColor;

void main()
{
  if ( isCheckerboard > 0 )
  {
    fragColor = max ( texture ( textureData, fWorldUV ), texture( checkerboardData, fPixelPatternUV ) );
  }
  else
  {
    fragColor = texture ( textureData, fWorldUV );
  }
}
