#include "CpuBulkHasher.hpp"

#include <openssl/evp.h>
#include <stdexcept>

std::array<std::uint64_t, 4> sha256_words_be(const std::string& input) {
  unsigned char hash[32];
  unsigned int out_len = 0;

  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  if (!ctx) {
    throw std::runtime_error("EVP_MD_CTX_new failed");
  }

  if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("EVP_DigestInit_ex failed");
  }

  if (EVP_DigestUpdate(ctx, input.data(), input.size()) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("EVP_DigestUpdate failed");
  }

  if (EVP_DigestFinal_ex(ctx, hash, &out_len) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("EVP_DigestFinal_ex failed");
  }

  EVP_MD_CTX_free(ctx);

  std::array<std::uint64_t, 4> words{};

  for (std::size_t i = 0; i < 4; ++i) {
    const std::size_t base = i * 8;
    words[i] = (static_cast<std::uint64_t>(hash[base + 0]) << 56) |
               (static_cast<std::uint64_t>(hash[base + 1]) << 48) |
               (static_cast<std::uint64_t>(hash[base + 2]) << 40) |
               (static_cast<std::uint64_t>(hash[base + 3]) << 32) |
               (static_cast<std::uint64_t>(hash[base + 4]) << 24) |
               (static_cast<std::uint64_t>(hash[base + 5]) << 16) |
               (static_cast<std::uint64_t>(hash[base + 6]) <<  8) |
                static_cast<std::uint64_t>(hash[base + 7]);
  }

  return words;
}

bool meetsDifficulty(const SHA256Hash& hash, int difficulty){

  const int shift = 256 - difficulty;

  const std::size_t wordIndex = 3 - ( shift / 64); //word with the final required leading zero (words are 0 index on MSW, 3 on LSW)
  const std::size_t bitIndex = (shift % 64);  //bit index of the final required leading 0 (0 index on LSB, 64 index on MSB)

  // any words greater than the target word should be 0.
  // i.e. if difficulty is 65 then the first 64 bits (0th word) should be 0
  for (std::size_t i = 0; i < wordIndex; i++){
    if (hash.value[i] > 0) {
      return false;
    }
  }

  //anything less than this threshold will be 0
  //i.e if difficulty check is >=4 than it must start with atleast 0000....
  //therefor anything less than 0001..... will have atleast 4 leading 0's
  const std::uint64_t wordThreshold = 0b1ULL << bitIndex;
  return  hash.value[wordIndex] < wordThreshold;

  // the words less significant than our wordIndex don't matter

}

void CpuBulkHasher::setHashes(std::vector<std::string> plainText)
{
  this->plainText = plainText;
  textSet = true;
}

bool CpuBulkHasher::compute(){
  if (!computeHash()){
    return false;
  }
  computeIsMoreDifficult();
  return true;
}

bool CpuBulkHasher::computeHash()
{
  if (!textSet){
    return false;
  }
  hashes = {};
  for (std::size_t i = 0; i < batchSize; i++){
    hashes.push_back(sha256_words_be(plainText.at(i)));
  }
  return true;
}

const std::vector<SHA256Hash> &CpuBulkHasher::getHashes()
{
  textSet = false;
  return hashes;
}

const std::string &CpuBulkHasher::getPlaintextAt(std::size_t index){
  return this->plainText.at(index);
}

void CpuBulkHasher::computeIsMoreDifficult()
{
  isMoreDifficult = {};
  for (std::size_t i = 0; i < batchSize; i++){
    isMoreDifficult.push_back(meetsDifficulty(hashes[i], targetDifficulty));
  }
}

const std::vector<bool> &CpuBulkHasher::getIsMoreDifficult()
{
  return isMoreDifficult;
}

void CpuBulkHasher::setNewDifficulty(int newTarget)
{
    this->targetDifficulty = newTarget;
}
