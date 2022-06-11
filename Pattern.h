#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

class Pattern
{
  using Bits = std::vector<bool>;
public:
  using Ptr = std::unique_ptr<Pattern>;

public:
  Pattern( const std::string& name, const uint32_t w, const uint32_t h, const std::vector<bool>& bits );

  bool at( const uint32_t x, const uint32_t y ) const;
  const std::string& name() const;
  uint32_t width() const;
  uint32_t height() const;

protected:
  Bits& bits();

private:
  std::string mName;
  uint32_t mWidth;
  uint32_t mHeight;
  Bits mBits;
};

class RandomPattern : public Pattern
{
  public:
    RandomPattern( const std::string& name, const uint32_t w, const uint32_t h );
};
