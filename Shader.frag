#version 450 core

uniform usampler2D textureData;
in vec2 fUV;

out uvec4 fragColor;

void main()
{
  fragColor = texture( textureData, fUV );
}
