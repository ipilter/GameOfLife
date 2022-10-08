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

  bool At( const uint32_t x, const uint32_t y ) const;
  const std::string& GetName() const;
  void SetName( const std::string& name );
  uint32_t Width() const;
  uint32_t Height() const;

  void Rotate();
  const float& GetRotation() const;

  void Write( std::ofstream& stream );
  void Read( std::ifstream& stream );

protected:
  Bits& GetBits();

private:
  std::string mName = "";
  uint32_t mWidth = 0;
  uint32_t mHeight = 0;
  Bits mBits;
  float mRotation = 0.0f;
};
