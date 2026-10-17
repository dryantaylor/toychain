#ifndef OPENCL_BULK_HASHER_HPP
#define OPENCL_BULK_HASHER_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 310

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <random>

#include <CL/opencl.hpp>

#include "BulkHasher.hpp"

class OpenCLBulkHasher : public BulkHasher {
public:
    OpenCLBulkHasher(std::size_t batchSize = 0, int startingDifficulty = 1);

    static std::string readFile(const std::string& path);

    void setString(std::string str) override;
    bool compute() override;

    const std::vector<SHA256Hash>& getHashes() override;
    const std::string& getPlaintextAt(std::size_t index) override;
    const std::vector<bool>& getIsMoreDifficult() override;
    void setNewDifficulty(int newTarget) override;
  private:
    std::uint64_t randomSeedLow();
    std::uint64_t randomSeedHigh();

private:
    std::mt19937_64 randomGen;
    std::uniform_int_distribution<std::uint64_t> distrib;
    
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel computeKernel;
    std::size_t localWorkSize;

    std::vector<std::string> plainText;
    std::vector<SHA256Hash> hashes;
    std::vector<bool> isMoreDifficult;
    int targetDifficulty;
    std::string stringPrefix;

};

#endif