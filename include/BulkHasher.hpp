#ifndef BULKHASHER_HPP
#define BULKHASHER_HPP
#include <cstdint>
#include <vector>
#include <string>
#include "SHA256Hash.hpp"

struct BulkHasher {
  public:
    BulkHasher(std::size_t batchSize) : batchSize(batchSize) {};
    virtual ~BulkHasher() = default;
    virtual void setHashes(const std::vector<std::string>& plainText) = 0;
    virtual bool computeHash() = 0;
    virtual const std::vector<SHA256Hash>& getHashes() = 0;

    virtual void computeIsMoreDifficult() = 0;
    virtual const std::vector<bool>& getIsMoreDifficult() = 0;
    virtual void setNewDifficulty(int newTarget) = 0;

    std::size_t getBatchSize() {return batchSize;};
  protected:
    std::size_t batchSize;
};
#endif