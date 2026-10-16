#ifndef OPENCL_BULK_HASHER_HPP
#define OPENCL_BULK_HASHER_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 310

#include <cstddef>
#include <string>
#include <vector>

#include <CL/opencl.hpp>

#include "BulkHasher.hpp"

class OpenCLBulkHasher : public BulkHasher {
public:
    OpenCLBulkHasher(std::size_t batchSize = 0, int startingDifficulty = 1);

    static std::string readFile(const std::string& path);

    void setHashes(std::vector<std::string> plainText) override;
    bool compute() override;

    const std::vector<SHA256Hash>& getHashes() override;
    const std::string& getPlaintextAt(std::size_t index) override;
    const std::vector<bool>& getIsMoreDifficult() override;
    void setNewDifficulty(int newTarget) override;

private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel computeKernel;
    std::vector<std::string> plainText;
    std::vector<SHA256Hash> hashes;
    std::vector<bool> isMoreDifficult;
    int targetDifficulty;
};

#endif