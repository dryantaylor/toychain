#ifndef CPU_BULKHASHER_HPP
#define CPU_BULKHASHER_HPP

#include "BulkHasher.hpp"
#include <vector>
#include <string>
#include <random>

class CpuBulkHasher : public BulkHasher {
  public:
    CpuBulkHasher(std::size_t batchSize, int startingDifficulty = 1) : BulkHasher(batchSize), targetDifficulty(startingDifficulty),
      randomGen(std::random_device{}()),
      distrib(0, INT_MAX)  {};
    void setString(std::string str) override;
    bool compute() override;
    bool computeHash();
    const std::vector<SHA256Hash>& getHashes() override;
    const std::string& getPlaintextAt(std::size_t index) override;

    void computeIsMoreDifficult();
    const std::vector<bool>& getIsMoreDifficult() override;
    void setNewDifficulty(int newTarget) override;

  private:
    std::mt19937 randomGen;
    std::uniform_int_distribution<int> distrib;

    std::string stringPrefix;
    std::vector<std::string> plainText;
    bool textSet = false;
    std::vector<SHA256Hash> hashes;
    std::vector<bool> isMoreDifficult;
    int targetDifficulty = 0;
};


#endif