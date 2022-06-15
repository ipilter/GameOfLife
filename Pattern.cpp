#pragma once

#include "Pattern.h"
#include "Math.h"
#include "Util.h"

Pattern::Pattern( const std::string& name, const uint32_t w, const uint32_t h, const std::vector<bool>& bits )
  : mName( name )
  , mWidth( w )
  , mHeight( h )
  , mBits( bits )
{}

Pattern::Pattern()
{}

bool Pattern::At( const uint32_t x, const uint32_t y ) const
{
  return mBits[x + mWidth * y];
}

const std::string& Pattern::GetName() const
{
  return mName;
}

void Pattern::SetName( const std::string& name )
{
  mName = name;
}

uint32_t Pattern::Width() const
{
  return mWidth;
}

uint32_t Pattern::Height() const
{
  return mHeight;
}

Pattern::Bits& Pattern::GetBits()
{
  return mBits;
}

void Pattern::Write( std::ofstream& stream )
{
  util::Write_t( stream, mWidth );
  util::Write_t( stream, mHeight );
  util::Write_t( stream, mName );
  util::Write_t( stream, mBits );
}

void Pattern::Read( std::ifstream& stream )
{
  util::Read_t<uint32_t>( stream, mWidth );
  util::Read_t<uint32_t>( stream, mHeight );
  util::Read_t<>( stream, mName );
  util::Read_t<>( stream, mBits );
}

void Pattern::Rotate()
{
  Pattern old( *this );
  mWidth = old.mHeight;
  mHeight = old.mWidth;
  mBits.resize( mWidth * mHeight );

  for ( uint32_t j = 0; j < mHeight; ++j )
  {
    for ( uint32_t i = 0; i < mWidth; ++i )
    {
      mBits[i + j * mWidth] = old.mBits[( old.mWidth - 1 - j ) + i * old.mWidth];
    }
  }
}
