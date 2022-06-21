#version 450 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 uv;

uniform mat4 vpMatrix;
out vec2 fUV;

void main()
{
  fUV = uv;
  gl_Position = vpMatrix * vec4( vertexPosition_modelspace, 1.0 );
}
