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

  void Bind();
  void Unbind();

  void BindTextureUnit( const uint32_t unitId );
  void UnbindTextureUnit();

  void CreateFromArray( const uint32_t* array );
  void CreateFromPBO();
  void UpdateFromPBO();
  void UpdateFromPBO( uint32_t regionPosX, uint32_t regionPosY, uint32_t regionWidth, uint32_t regionHeight );

  uint32_t Width() const;
  uint32_t Height() const;

private:
  uint32_t mId = 0;
  uint32_t mWidth = 0;
  uint32_t mHeight = 0;
};
