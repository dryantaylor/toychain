#include "OpenCLBulkHasher.hpp"

#include <algorithm>
#include <cctype>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>


int GenerateBatchSizeEstimate(cl::Device& device, cl_ulong op_write_size) {
    cl_uint computeUnits = 0;
    size_t maxWorkGroupSize = 0;
    cl_ulong maxAllocBytes = 0;
    cl_ulong maxClockMHz = 0;
    cl_ulong globalMemBytes = 0;
    cl_device_type deviceType = 0;
    std::string deviceName;

    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &computeUnits);
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
    device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxAllocBytes);
    device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &maxClockMHz);
    device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemBytes);
    device.getInfo(CL_DEVICE_TYPE, &deviceType);
    device.getInfo(CL_DEVICE_NAME, &deviceName);

    std::string lowerName = deviceName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    const bool isIntegratedGpu = (deviceType & CL_DEVICE_TYPE_GPU) &&
        (computeUnits <= 12U || maxClockMHz < 1500U ||
         lowerName.find("intel") != std::string::npos
        )
        ;

    const size_t preferredWorkGroup = std::min<size_t>(256U, std::max<size_t>(64U, maxWorkGroupSize));

    size_t targetItems = 0;
    if (isIntegratedGpu) {
        targetItems = std::max<size_t>(preferredWorkGroup * 4096U,
                                      static_cast<size_t>(computeUnits) * 2U * 1024U * 1024U);
        targetItems = std::min<size_t>(targetItems, 32U * 1024U * 1024U);
    } else {
        targetItems = std::max<size_t>(64U * 1024U * 1024U,
                                      static_cast<size_t>(computeUnits) * 8U * 1024U * 1024U);
        targetItems = std::min<size_t>(targetItems, 256U * 1024U * 1024U);
      }

    size_t estimated = targetItems;

    const size_t memoryBound = static_cast<size_t>(
        maxAllocBytes / std::max<cl_ulong>(op_write_size, 1ULL));
    if (memoryBound > 0) {
        estimated = std::min(estimated, memoryBound);
    }

    const size_t totalMemWindow = static_cast<size_t>(globalMemBytes / 8U);
    if (totalMemWindow > 0) {
        estimated = std::min(estimated, totalMemWindow);
    }

    estimated = std::max<size_t>(preferredWorkGroup * 128U, estimated);
    estimated = ((estimated + preferredWorkGroup - 1U) / preferredWorkGroup) * preferredWorkGroup;

    if (estimated == 0) {
        return static_cast<int>(preferredWorkGroup * 128U);
    }

    return static_cast<int>(estimated);
}

OpenCLBulkHasher::OpenCLBulkHasher(std::size_t batchSize, int startingDifficulty)
    : BulkHasher(), localWorkSize(128), plainText(), targetDifficulty(startingDifficulty),
    randomGen(std::random_device{}()),
    distrib(0, std::numeric_limits<std::uint64_t>::max()) {
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
    size_t maxWorkGroupSize = 128;
    cl_device_type deviceType = 0;
    std::string deviceName;
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
    device.getInfo(CL_DEVICE_TYPE, &deviceType);
    device.getInfo(CL_DEVICE_NAME, &deviceName);

    std::string lowerName = deviceName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    const bool isIntegrated = (deviceType & CL_DEVICE_TYPE_GPU) &&
        (lowerName.find("intel") != std::string::npos ||
         lowerName.find("uhd") != std::string::npos ||
         lowerName.find("iris") != std::string::npos ||
         lowerName.find("radeon graphics") != std::string::npos ||
         lowerName.find("vega") != std::string::npos ||
         lowerName.find("integrated") != std::string::npos);

    this->localWorkSize = isIntegrated ? std::min<std::size_t>(std::max<std::size_t>(1, maxWorkGroupSize), 128U)
                                      : std::min<std::size_t>(std::max<std::size_t>(1, maxWorkGroupSize), 256U);

    if (batchSize == 0) {
        constexpr cl_ulong inputSizeApprox = 0;
        constexpr cl_ulong outputSize = 5 * sizeof(std::uint64_t) + sizeof(bool);
        this->batchSize = GenerateBatchSizeEstimate(device, outputSize);
        this->batchSize = std::max<std::size_t>(this->batchSize, this->localWorkSize * 64U);
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

void OpenCLBulkHasher::setString(std::string str)
{
  this->stringPrefix = str;
}

bool OpenCLBulkHasher::compute() {
  const std::uint64_t seedLow = randomSeedLow();
  const std::uint64_t seedHigh = randomSeedHigh();

  this->plainText.clear();
  this->hashes.clear();
  this->isMoreDifficult.clear();

  std::vector<long> winnerNonces(batchSize);
  std::vector<std::uint64_t> winnerHashes(batchSize * 4U);
  int matchCount = 0;

  cl::Buffer stringPrefixBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                stringPrefix.size() + 1U, stringPrefix.data());
  cl::Buffer winnerNoncesBuffer(context, CL_MEM_READ_WRITE,
                               batchSize * sizeof(long));
  cl::Buffer winnerHashesBuffer(context, CL_MEM_READ_WRITE,
                               batchSize * 4U * sizeof(std::uint64_t));
  cl::Buffer matchCountBuffer(context, CL_MEM_READ_WRITE,
                             sizeof(int));

  cl_int argErr = computeKernel.setArg(0, seedLow);
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(0) failed with code " << argErr << std::endl;
      return false;
  }

  argErr = computeKernel.setArg(1, seedHigh);
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(1) failed with code " << argErr << std::endl;
      return false;
  }

  argErr = computeKernel.setArg(2, stringPrefixBuffer);
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(2) failed with code " << argErr << std::endl;
      return false;
  }

  argErr = computeKernel.setArg(3, static_cast<int>(stringPrefix.length()));
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(3) failed with code " << argErr << std::endl;
      return false;
  }

  argErr = computeKernel.setArg(4, winnerNoncesBuffer);
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(4) failed with code " << argErr << std::endl;
      return false;
  }

  argErr = computeKernel.setArg(5, winnerHashesBuffer);
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(5) failed with code " << argErr << std::endl;
      return false;
  }

  argErr = computeKernel.setArg(6, matchCountBuffer);
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(6) failed with code " << argErr << std::endl;
      return false;
  }

  argErr = computeKernel.setArg(7, targetDifficulty);
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(7) failed with code " << argErr << std::endl;
      return false;
  }

  argErr = computeKernel.setArg(8, static_cast<int>(batchSize));
  if (argErr != CL_SUCCESS) {
      std::cerr << "OpenCL setArg(8) failed with code " << argErr << std::endl;
      return false;
  }

  cl_int writeErr = queue.enqueueWriteBuffer(matchCountBuffer, CL_TRUE, 0, sizeof(int), &matchCount);
  if (writeErr != CL_SUCCESS) {
      std::cerr << "OpenCL write match count failed with code " << writeErr << std::endl;
      return false;
  }

  const size_t globalSize = ((batchSize + localWorkSize - 1U) / localWorkSize) * localWorkSize;
  cl_int kernelErr = queue.enqueueNDRangeKernel(
      computeKernel,
      cl::NullRange,
      cl::NDRange(globalSize),
      cl::NDRange(localWorkSize));
  if (kernelErr != CL_SUCCESS) {
      std::cerr << "OpenCL kernel launch failed with code " << kernelErr << std::endl;
      return false;
  }

  cl_int readErr = queue.enqueueReadBuffer(matchCountBuffer, CL_TRUE, 0, sizeof(int), &matchCount);
  if (readErr != CL_SUCCESS) {
      std::cerr << "OpenCL read match count failed with code " << readErr << std::endl;
      return false;
  }

  if (matchCount > 0) {
      winnerNonces.resize(static_cast<std::size_t>(matchCount));
      winnerHashes.resize(static_cast<std::size_t>(matchCount) * 4U);

      readErr = queue.enqueueReadBuffer(winnerNoncesBuffer, CL_TRUE, 0,
                                       static_cast<std::size_t>(matchCount) * sizeof(long),
                                       winnerNonces.data());
      if (readErr != CL_SUCCESS) {
          std::cerr << "OpenCL read winners failed with code " << readErr << std::endl;
          return false;
      }

      readErr = queue.enqueueReadBuffer(winnerHashesBuffer, CL_TRUE, 0,
                                       static_cast<std::size_t>(matchCount) * 4U * sizeof(std::uint64_t),
                                       winnerHashes.data());
      if (readErr != CL_SUCCESS) {
          std::cerr << "OpenCL read winner hashes failed with code " << readErr << std::endl;
          return false;
      }
  }

  this->isMoreDifficult.resize(static_cast<std::size_t>(matchCount), true);
  this->plainText.reserve(static_cast<std::size_t>(matchCount));
  this->hashes.reserve(static_cast<std::size_t>(matchCount));

  for (int i = 0; i < matchCount; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 4U;
      this->plainText.emplace_back(stringPrefix + std::to_string(winnerNonces[static_cast<std::size_t>(i)]));
      this->hashes.emplace_back(SHA256Hash(winnerHashes[base], winnerHashes[base + 1U],
                                          winnerHashes[base + 2U], winnerHashes[base + 3U]));
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

std::uint64_t OpenCLBulkHasher::randomSeedLow()
{
  return distrib(randomGen);
}

std::uint64_t OpenCLBulkHasher::randomSeedHigh()
{
  return distrib(randomGen);
}
