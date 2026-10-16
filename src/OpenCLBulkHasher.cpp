#include "OpenCLBulkHasher.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

int GenerateBatchSizeEstimate(cl::Device& device, cl_ulong op_read_size, cl_ulong op_write_size) {
    cl_uint computeUnits = 0;
    size_t maxWorkGroupSize = 0;
    cl_ulong globalMemBytes = 0;
    cl_ulong maxAllocBytes = 0;

    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &computeUnits);
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
    device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemBytes);
    device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxAllocBytes);

    const size_t preferredWorkGroup = std::min<size_t>(128, std::max<size_t>(1, maxWorkGroupSize));
    size_t estimated = preferredWorkGroup * static_cast<size_t>(computeUnits) * 8;

    const size_t memoryBound = static_cast<size_t>(
        std::min<cl_ulong>(globalMemBytes / op_read_size, maxAllocBytes / op_write_size)
    );

    if (memoryBound > 0) {
        estimated = std::min(estimated, memoryBound);
    }

    estimated = std::max<size_t>(preferredWorkGroup, estimated);
    estimated = (estimated / preferredWorkGroup) * preferredWorkGroup;

    if (estimated == 0) {
        return static_cast<int>(preferredWorkGroup);
    }

    return static_cast<int>(estimated);
}

bool meetsDifficultyCL(const SHA256Hash& hash, int difficulty) {
    const int shift = 256 - difficulty;
    const std::size_t wordIndex = 3 - (shift / 64);
    const std::size_t bitIndex = shift % 64;

    for (std::size_t i = 0; i < wordIndex; i++) {
        if (hash.value[i] > 0) {
            return false;
        }
    }

    const std::uint64_t wordThreshold = 0b1ULL << bitIndex;
    return hash.value[wordIndex] < wordThreshold;
}

OpenCLBulkHasher::OpenCLBulkHasher(std::size_t batchSize, int startingDifficulty)
    : BulkHasher(), plainText(), targetDifficulty(startingDifficulty) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No Platforms found");
    }

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No GPU's found");
    }

    cl::Device device = devices[0];

    if (batchSize == 0) {
        constexpr cl_ulong inputSizeApprox = (256 + 12 + 20) * sizeof(unsigned char);
        constexpr cl_ulong outputSize = 4 * sizeof(std::uint64_t);
        this->batchSize = GenerateBatchSizeEstimate(device, inputSizeApprox, outputSize);
        std::cout << "Estimated batch size of: " << this->batchSize << std::endl;
    } else {
        this->batchSize = batchSize;
    }

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    std::string hash_source = readFile("kernels/generate_hash.cl");
    std::string difficulty_source = readFile("kernels/calculate_difficulty.cl");
    std::string main_source = readFile("kernels/main.cl");
    std::cout << "Generating Program" << std::endl;

    cl::Program program(context, hash_source + "\n" + difficulty_source+"\n"+main_source);
    cl_int buildResult = program.build(device);
    if (buildResult != CL_SUCCESS) {
        std::cerr << "OpenCL build failed with code " << buildResult << "\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        throw std::runtime_error("OpenCL kernel build failed");
    }

    std::cout << "setting kernal" << std::endl;
    computeKernel = cl::Kernel(program, "compute");
    std::cout << "Compute Kernel handle      : " << computeKernel() << std::endl;
}

std::string OpenCLBulkHasher::readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open OpenCL kernel file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void OpenCLBulkHasher::setHashes(std::vector<std::string> plainText) {
    this->plainText = std::move(plainText);
}

bool OpenCLBulkHasher::compute()
{
  if (plainText.empty()){
    return false;
  }

  const std::size_t workItems = std::min<std::size_t>(batchSize, plainText.size());
    if (workItems == 0) {
        return false;
    }

    hashes.clear();
    hashes.resize(workItems);

    std::vector<unsigned char> flatInput;
    std::vector<int> inputLengths;
    std::vector<int> inputOffsets;
    flatInput.reserve(workItems * 64U);

    std::size_t totalBytes = 0;
    for (const std::string& text : plainText) {
        inputOffsets.push_back(static_cast<int>(totalBytes));
        inputLengths.push_back(static_cast<int>(text.size()));
        flatInput.insert(flatInput.end(), text.begin(), text.end());
        totalBytes += text.size();
    }

    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          flatInput.size() * sizeof(unsigned char), flatInput.data());
    cl::Buffer lengthsBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            inputLengths.size() * sizeof(int), inputLengths.data());
    cl::Buffer offsetsBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            inputOffsets.size() * sizeof(int), inputOffsets.data());
    cl::Buffer outputHashesBuffer(context, CL_MEM_READ_WRITE,
                           workItems * 4U * sizeof(std::uint64_t));
    cl::Buffer outputDifficultyBuffer(context, CL_MEM_WRITE_ONLY, workItems*sizeof(bool));

    cl_int argErr = computeKernel.setArg(0, inputBuffer);
    if (argErr != CL_SUCCESS) {
        std::cerr << "OpenCL setArg(0) failed with code " << argErr << std::endl;
        return false;
    }
    argErr = computeKernel.setArg(1, lengthsBuffer);
    if (argErr != CL_SUCCESS) {
        std::cerr << "OpenCL setArg(1) failed with code " << argErr << std::endl;
        return false;
    }
    argErr = computeKernel.setArg(2, offsetsBuffer);
    if (argErr != CL_SUCCESS) {
        std::cerr << "OpenCL setArg(2) failed with code " << argErr << std::endl;
        return false;
    }
    argErr = computeKernel.setArg(3, outputHashesBuffer);
    if (argErr != CL_SUCCESS) {
        std::cerr << "OpenCL setArg(3) failed with code " << argErr << std::endl;
        return false;
    }
    argErr = computeKernel.setArg(4, outputDifficultyBuffer);
    if (argErr != CL_SUCCESS) {
        std::cerr << "OpenCL setArg(4) failed with code " << argErr << std::endl;
        return false;
    }
    int effectiveDifficulty = std::max(0, std::min(256, targetDifficulty));

    argErr = computeKernel.setArg(5, effectiveDifficulty);
    if (argErr != CL_SUCCESS) {
        std::cerr << "OpenCL setArg(5) failed with code " << argErr << std::endl;
        return false;
    }

    argErr = computeKernel.setArg(6, static_cast<int>(workItems));
    if (argErr != CL_SUCCESS) {
        std::cerr << "OpenCL setArg(6) failed with code " << argErr << std::endl;
        return false;
    }

    cl_int kernelErr = queue.enqueueNDRangeKernel(computeKernel, cl::NullRange, cl::NDRange(workItems), cl::NullRange);
    if (kernelErr != CL_SUCCESS) {
        std::cerr << "OpenCL kernel launch failed with code " << kernelErr << std::endl;
        return false;
    }

    std::vector<std::uint64_t> HashesOutput(workItems * 4U);
    cl_int readErr = queue.enqueueReadBuffer(outputHashesBuffer, CL_TRUE, 0,
                                            HashesOutput.size() * sizeof(std::uint64_t), HashesOutput.data());
    if (readErr != CL_SUCCESS) {
        std::cerr << "OpenCL read failed with code " << readErr << std::endl;
        return false;
    }
    bool outputDifficulty[workItems];
    readErr = queue.enqueueReadBuffer(outputDifficultyBuffer,CL_TRUE,0, workItems * sizeof(bool), outputDifficulty);
    if (readErr != CL_SUCCESS) {
        std::cerr << "OpenCL read failed with code " << readErr << std::endl;
        return false;
    }

    isMoreDifficult.clear();
    for (std::size_t i = 0; i < workItems; ++i) {
        const std::size_t base = i * 4U;
        hashes[i] = SHA256Hash(HashesOutput[base], HashesOutput[base + 1U], HashesOutput[base + 2U], HashesOutput[base + 3U]);
        isMoreDifficult.push_back(outputDifficulty[i]);
    }
    
    return true;


}


const std::vector<SHA256Hash>& OpenCLBulkHasher::getHashes() {
    return hashes;
}

const std::string& OpenCLBulkHasher::getPlaintextAt(std::size_t index) {
    return this->plainText.at(index);
}

const std::vector<bool>& OpenCLBulkHasher::getIsMoreDifficult() {
    return isMoreDifficult;
}

void OpenCLBulkHasher::setNewDifficulty(int newTarget) {
    targetDifficulty = newTarget;
}
