#ifndef TEST_OPENCL_HPP
#define TEST_OPENCL_HPP
/*
#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

inline void checkOpenCL(cl_int err, const std::string& message) {
  if (err != CL_SUCCESS) {
    throw std::runtime_error(message + " (OpenCL error code: " + std::to_string(err) + ")");
  }
}

int GenerateSensibleBatchSize(cl::Device& device, cl_ulong op_read_size, cl_ulong op_write_size) {
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

int test() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    std::cerr << "No OpenCL platform found." << std::endl;
    return 1;
  }

  cl::Device device;
  std::vector<cl::Device> devices;
  platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (devices.empty()) {
    std::cerr << "No OpenCL device found." << std::endl;
    return 1;
  }
  device = devices[0];
  std::string name;
  device.getInfo(CL_DEVICE_NAME, &name);
  std::cout << "Using Device: " << name << std::endl;


  const int N = GenerateSensibleBatchSize(device, (sizeof(float)*3), sizeof(float));
  std::cout << "Suggested Batch size of " << N << std::endl;
  std::vector<float> A(N);
  std::vector<float> B(N);
  std::vector<float> C(N);

  for (int i = 0; i < N; ++i) {
    A[i] = static_cast<float>(i + 1);
    B[i] = static_cast<float>((i + 1) * 2);
  }



  cl::Context context(device);
  cl::CommandQueue queue(context, device, 0);

  cl::Buffer dA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * A.size(), A.data());
  cl::Buffer dB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * B.size(), B.data());
  cl::Buffer dC(context, CL_MEM_WRITE_ONLY,
                sizeof(float) * C.size());

  const char* kernelSource =
    "__kernel void add_vectors(__global const float* A, __global const float* B, __global float* C, const int N) {\n"
    "  int i = get_global_id(0);\n"
    "  if (i < N) {\n"
    "    C[i] = A[i] + B[i];\n"
    "  }\n"
    "}\n";

  cl::Program program(context, kernelSource);
  cl_int buildErr = program.build({device});
  if (buildErr != CL_SUCCESS) {
    std::cerr << "OpenCL build failed:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    return 1;
  }

  cl::Kernel kernel(program, "add_vectors");
  kernel.setArg(0, dA);
  kernel.setArg(1, dB);
  kernel.setArg(2, dC);
  kernel.setArg(3, N);

  cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to launch OpenCL kernel" << std::endl;
    return 1;
  }

  err = queue.enqueueReadBuffer(dC, CL_TRUE, 0, sizeof(float) * C.size(), C.data());
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to read result back from GPU" << std::endl;
    return 1;
  }

  std::cout << "C = A + B on GPU:\n";
  for (int i = 0; i < N; ++i) {
    //std::cout << "C[" << i << "] = " << C[i] << std::endl;
  }

  return 0;
}
  */
#endif
