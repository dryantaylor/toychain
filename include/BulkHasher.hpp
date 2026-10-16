#ifndef BULKHASHER_HPP
#define BULKHASHER_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "SHA256Hash.hpp"

struct BulkHasher {
public:
    BulkHasher(std::size_t batchSize);
    BulkHasher();
    virtual ~BulkHasher() = default;

    virtual void setHashes(std::vector<std::string> plainText) = 0;
    virtual bool computeHash() = 0;
    virtual const std::vector<SHA256Hash>& getHashes() = 0;
    virtual const std::string& getPlaintextAt(std::size_t index) = 0;

    virtual void computeIsMoreDifficult() = 0;
    virtual const std::vector<bool>& getIsMoreDifficult() = 0;
    virtual void setNewDifficulty(int newTarget) = 0;

    std::size_t getBatchSize();

protected:
    std::size_t batchSize;
};

#endif