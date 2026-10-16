#ifndef SHA256_HASH_HPP
#define SHA256_HASH_HPP

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>

class SHA256Hash {
public:
    SHA256Hash(std::uint64_t w0, std::uint64_t w1, std::uint64_t w2, std::uint64_t w3);
    SHA256Hash(const std::array<std::uint64_t, 4>& value);
    SHA256Hash(const std::string& str);
    SHA256Hash();

    std::string to_string() const;
    const std::uint64_t& w0() const;
    const std::uint64_t& w1() const;
    const std::uint64_t& w2() const;
    const std::uint64_t& w3() const;

    std::uint64_t value[4];
    int difficulty() const;
};

std::ostream& operator<<(std::ostream& os, const SHA256Hash& hash);

#endif