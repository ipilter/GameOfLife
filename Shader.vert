#version 450 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 worldUV;
layout(location = 2) in vec2 pixelPatternUV;

uniform mat4 vpMatrix;
out vec2 fWorldUV;
out vec2 fPixelPatternUV;

void main()
{
  fWorldUV = worldUV;
  fPixelPatternUV = pixelPatternUV;
  gl_Position = vpMatrix * vec4(vertexPosition_modelspace, 1.0);
}
