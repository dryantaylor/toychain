#include "SHA256Hash.hpp"
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

SHA256Hash::SHA256Hash(std::uint64_t w0, std::uint64_t w1, std::uint64_t w2, std::uint64_t w3)
{
  this->value[0] = w0;
  this->value[1] = w1;
  this->value[2] = w2;
  this->value[3] = w3;
}

SHA256Hash::SHA256Hash(const std::array<std::uint64_t, 4> &value)
{
  this->value[0] = value[0];
  this->value[1] = value[1];
  this->value[2] = value[2];
  this->value[3] = value[3];
}

SHA256Hash::SHA256Hash(const std::string &str)
{
  std::string normalized = str;

  while (!normalized.empty() && std::isspace(static_cast<unsigned char>(normalized.front()))) {
    normalized.erase(normalized.begin());
  }
  while (!normalized.empty() && std::isspace(static_cast<unsigned char>(normalized.back()))) {
    normalized.pop_back();
  }

  if (normalized.size() >= 2 && normalized[0] == '0' && (normalized[1] == 'x' || normalized[1] == 'X')) {
    normalized.erase(0, 2);
  }

  if (normalized.size() != 64) {
    throw std::invalid_argument("SHA256Hash string must contain exactly 64 hex characters");
  }

  for (std::size_t i = 0; i < 4; ++i) {
    const std::string chunk = normalized.substr(i * 16, 16);
    std::size_t parsedChars = 0;
    const std::uint64_t parsed = static_cast<std::uint64_t>(std::stoull(chunk, &parsedChars, 16));

    if (parsedChars != chunk.size()) {
      throw std::invalid_argument("SHA256Hash string contains non-hex characters");
    }

    value[i] = parsed;
  }
}

SHA256Hash::SHA256Hash()
{
  //This function exists to allow implicit initialisation of std::array<SHA256Hash, n>
  value[0] = 0ULL;
  value[1] = 0ULL;
  value[2] = 0ULL;
  value[3] = 0ULL;
}

std::string SHA256Hash::to_string() const
{
  std::ostringstream oss;
  oss << *this;
  return oss.str();
}

int SHA256Hash::difficulty() const
{
  int leadingZeroBits = 0;

  for (std::size_t wordIndex = 0; wordIndex < 4; ++wordIndex) {
    const std::uint64_t word = value[wordIndex];

    for (int bitIndex = 63; bitIndex >= 0; --bitIndex) {
      if ((word >> bitIndex) & 1ULL) {
        return leadingZeroBits;
      }
      ++leadingZeroBits;
    }
  }

  return 256;
}

std::ostream &operator<<(std::ostream &os, const SHA256Hash &hash)
{
  std::ios_base::fmtflags currentFlags = os.flags();
  os << std::hex << std::setw(16) << std::setfill('0') << hash.w0() 
     << std::setw(16) << std::setfill('0') << hash.w1() 
     << std::setw(16) << std::setfill('0') << hash.w2() 
     << std::setw(16) << std::setfill('0') << hash.w3();
  os.setf(currentFlags);
  return os;
}
