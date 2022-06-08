#pragma once

#include <cstdint>
#include <vector>
#include <memory>

// pattern storage
class Pattern
{
public:
  using Ptr = std::unique_ptr<Pattern>;

public:
  Pattern( const uint32_t w, const uint32_t h, const std::vector<bool>& bits );

  bool at( const uint32_t x, const uint32_t y ) const;

  uint32_t width() const;
  uint32_t height() const;

private:
  uint32_t mWidth;
  uint32_t mHeight;
  std::vector<bool> mBits;
};
