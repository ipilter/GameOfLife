#pragma once

#include <gl/glew.h>

// pattern storage
struct Pattern
{
  Pattern( GLuint w, GLuint h, const std::vector<bool>& bits )
    : mWidth( w )
    , mHeight( h )
    , mBits( bits )
  {}

  bool at( const GLuint x, const GLuint y ) const
  {
    return mBits[x + mWidth * y];
  }

  GLuint width() const
  {
    return mWidth;
  }

  GLuint height() const
  {
    return mHeight;
  }

private:
  GLuint mWidth;
  GLuint mHeight;
  std::vector<bool> mBits;
};
