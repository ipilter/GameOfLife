#pragma once

#include "Pattern.h"
#include "Math.h"

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

Pattern::Bits& Pattern::bits()
{
  return mBits;
}

RandomPattern::RandomPattern( const std::string& name, const uint32_t w, const uint32_t h )
  : Pattern( name, w, h, std::vector<bool>( w * h, 0) )
{
  size_t rw = w;
  size_t rh = h;

  std::vector<bool>& rp = bits();
  for ( auto h = 0ull; h < rh; ++h )
  {
    for ( auto w = 0ull; w < rw; ++w )
    {
      auto r = math::random();
      rp[w + rw * h] = r > 0.8f;
    }
  }
}
