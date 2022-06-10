#pragma once

#include "Pattern.h"

Pattern::Pattern( const std::string& name, const uint32_t w, const uint32_t h, const std::vector<bool>& bits )
  : mName( name )
  , mWidth( w )
  , mHeight( h )
  , mBits( bits )
{}

bool Pattern::at( const uint32_t x, const uint32_t y ) const
{
  return mBits[x + mWidth * y];
}

const std::string& Pattern::name() const
{
  return mName;
}

uint32_t Pattern::width() const
{
  return mWidth;
}

uint32_t Pattern::height() const
{
  return mHeight;
}
