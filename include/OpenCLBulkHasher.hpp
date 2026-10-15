#ifndef OPENCL_BULK_HASHER_HPP
#define OPENCL_BULK_HASHER_HPP

#include "BulkHasher.hpp"
#include <CL/cl2.hpp>

class OpenCLBulkHasher : public BulkHasher {
public:
    OpenCLBulkHasher(std::size_t batchSize, int startingDifficulty = 1)
        : BulkHasher(batchSize), targetDifficulty(startingDifficulty) {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        auto platform = platforms.front();
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

        context = cl::Context(devices);
        queue = cl::CommandQueue(context, devices[0]);

        std::string source = R"(
            __kernel void check_hashes(__global const uchar* input,
                                      __global int* output) {
                int gid = get_global_id(0);
                // compute hash and difficulty here
                output[gid] = 1; // or 0
            }
        )";

        cl::Program program(context, source);
        program.build(devices);
        kernel = cl::Kernel(program, "check_hashes");
    }

    void setHashes(const std::vector<std::string>& plainText) override {
        this->plainText = plainText;
    }

    bool computeHash() override {
        // allocate buffers, enqueue kernel, read results
        return true;
    }

    const std::vector<SHA256Hash>& getHashes() override { return hashes; }
    void computeIsMoreDifficult() override {}
    const std::vector<bool>& getIsMoreDifficult() override { return isMoreDifficult; }
    void setNewDifficulty(int newTarget) override { targetDifficulty = newTarget; }

private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    std::vector<std::string> plainText;
    std::vector<SHA256Hash> hashes;
    std::vector<bool> isMoreDifficult;
    int targetDifficulty;
};

#endif