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
  Pattern();
  Pattern( const std::string& name, const uint32_t w, const uint32_t h, const std::vector<bool>& bits );

  bool at( const uint32_t x, const uint32_t y ) const;
  const std::string& name() const;
  uint32_t width() const;
  uint32_t height() const;

  void write( std::ofstream& stream );
  void read( std::ifstream& stream );

  void rotate();
  void setName( const std::string& name );

protected:
  Bits& bits();

private:
  std::string mName = "";
  uint32_t mWidth = 0;
  uint32_t mHeight = 0;
  Bits mBits;
};
