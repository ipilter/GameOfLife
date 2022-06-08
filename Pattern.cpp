#pragma once

#include "Pattern.h"

Pattern::Pattern( const uint32_t w, const uint32_t h, const std::vector<bool>& bits )
  : mWidth( w )
  , mHeight( h )
  , mBits( bits )
{}

bool Pattern::at( const uint32_t x, const uint32_t y ) const
{
  return mBits[x + mWidth * y];
}

uint32_t Pattern::width() const
{
  return mWidth;
}

uint32_t Pattern::height() const
{
  return mHeight;
}
