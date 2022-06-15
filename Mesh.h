#pragma once

#include <cstdint>
#include <vector>
#include <memory>

class Mesh
{
public:
  using Ptr = std::unique_ptr<Mesh>;

public:
  Mesh( const std::vector<float>& points, const std::vector<uint32_t>& indices, const uint32_t stride, const uint32_t vertexOffset, const uint32_t worldTexelOffset, const size_t patternTexelOffset );
  ~Mesh();

  void Bind();
  void Unbind();
  void Render();

public:
  uint32_t mVbo = 0;
  uint32_t mIbo = 0;
  uint32_t mVao = 0;
  uint32_t mIndexCount = 0;;
};
