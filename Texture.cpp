#include <gl/glew.h>

#include "Texture.h"

Texture::Texture( const uint32_t width, const uint32_t height )
  : mWidth( width )
  , mHeight( height )
{
  glGenTextures( 1, &mId );
  glBindTexture( GL_TEXTURE_2D, mId );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
  glBindTexture( GL_TEXTURE_2D, 0 );
}

Texture::~Texture()
{
  glDeleteTextures( 1, &mId );
}

void Texture::bind()
{
  glBindTexture( GL_TEXTURE_2D, mId );
}

void Texture::unbind()
{
  glBindTexture( GL_TEXTURE_2D, 0 );
}

void Texture::bindTextureUnit( const uint32_t unitId )
{
  glBindTextureUnit( unitId, mId );
}

void Texture::unbindTextureUnit()
{
  glBindTextureUnit( 0, 0 );
}

void Texture::createFromPBO()
{
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
}

void Texture::updateFromPBO()
{
  glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
}

void Texture::updateFromPBO( uint32_t regionPosX, uint32_t regionPosY, uint32_t regionWidth, uint32_t regionHeight )
{
  glPixelStorei( GL_UNPACK_ROW_LENGTH, mWidth );
  glTexSubImage2D( GL_TEXTURE_2D, 0
                   , regionPosX, regionPosY
                   , regionWidth, regionHeight
                   , GL_RGBA, GL_UNSIGNED_BYTE
                   , reinterpret_cast<void*>( regionPosX * 4ull + regionPosY * 4ull * mWidth ) );
  glPixelStorei( GL_UNPACK_ROW_LENGTH, 0 );
}

uint32_t Texture::width() const
{
  return mWidth;
}

uint32_t Texture::height() const
{
  return mHeight;
}
