#pragma once

#include <cstdint>
#include <memory>

class Texture
{
public:
  using Ptr = std::unique_ptr<Texture>;

public:
  Texture( const uint32_t width, const uint32_t height, const uint32_t wrap = GL_CLAMP );
  ~Texture();

  void bind();
  void unbind();

  void bindTextureUnit( const uint32_t unitId );
  void unbindTextureUnit();

  void createFromPBO();
  void updateFromPBO();
  void updateFromPBO( uint32_t regionPosX, uint32_t regionPosY, uint32_t regionWidth, uint32_t regionHeight );

  uint32_t width() const;
  uint32_t height() const;

private:
  uint32_t mId = 0;
  uint32_t mWidth = 0;
  uint32_t mHeight = 0;
};
