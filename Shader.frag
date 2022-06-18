#version 450 core

uniform sampler2D textureData;

in vec2 fUV;

out vec4 fragColor;

void main()
{
  fragColor = texture ( textureData, fUV );
}
