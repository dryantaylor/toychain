#include "BulkHasher.hpp"

BulkHasher::BulkHasher(std::size_t batchSize) : batchSize(batchSize) {}

BulkHasher::BulkHasher() : batchSize(0) {}

std::size_t BulkHasher::getBatchSize() {
    return batchSize;
}
