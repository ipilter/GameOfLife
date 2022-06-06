#version 450 core

uniform sampler2D textureData;

in vec2 uv;

out vec4 fragColor;

void main()
{
  fragColor = texture(textureData, uv);
}
