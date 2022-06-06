#version 450 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 vertexUV;

uniform mat4 vpMatrix;
out vec2 uv;

void main()
{
  uv = vertexUV;
  gl_Position = vpMatrix * vec4(vertexPosition_modelspace, 1.0);
}
