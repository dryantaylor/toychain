#include "CpuBulkHasher.hpp"
#include "OpenCLBulkHasher.hpp"
#include "SearchControl.hpp"
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>

std::map<std::string, std::string> parseArgs(int argc, char** argv){
  std::vector<std::string> args;
  for (int i= 1; i < argc; i++){
    args.push_back(argv[i]);
  }


  std::map<std::string, std::string> argMap = {{"name", ""}, {"hash",""}, {"batch", "0"}, {"file",""},
  {"hashCore","CPU"}};
  for (std::size_t i = 0; i < args.size()-1; i++){
    if (args.at(i) == "--name" || args.at(i) == "-n"){
      argMap.at("name") = args.at(i+1);
    }
    if (args.at(i) == "--hash" || args.at(i) == "-h"){
      argMap.at("hash") = args.at(i+1);
    }
    if (args.at(i) == "--file" || args.at(i) == "-f"){
      argMap.at("file") = args.at(i+1);
    }
    if (args.at(i) == "--batchSize" || args.at(i) == "-b"){
      argMap.at("batch") = args.at(i+1);
    }

    if (args.at(i) == "--hashCore" || args.at(i) == "-hc"){
      argMap.at("hashCore") = args.at(i+1);
    }
  }
  return argMap;
}

std::unique_ptr<BulkHasher> selectHashingCore(const std::string& coreName, std::size_t batchSize, int difficulty){
  if (coreName == "CPU"){
    if (batchSize == 0){
      batchSize = 1;
    }
    return std::make_unique<CpuBulkHasher>(batchSize, difficulty);
  }

  if (coreName == "OpenCL"){
    return std::make_unique<OpenCLBulkHasher>(batchSize, difficulty);
  }
  throw std::runtime_error("Invalid hashing Core selected");
}


int main(int argc, char** argv) {

  auto args = parseArgs(argc, argv);
  SHA256Hash startingHash = args.at("hash");
  int difficulty = startingHash.difficulty()+1;
  std::string name = args.at("name");
  std::size_t batchSize = std::stoi(args.at("batch"));
  bool usesCout = args.at("file") == "";
  std::ofstream* outFile;
  if (!usesCout) {
    outFile = new std::ofstream(args.at("file"));
  }

  auto searcher = SearchControl(
    std::move(selectHashingCore(args.at("hashCore"), batchSize, difficulty)),
    (usesCout ? std::cout : *outFile),
    name,
    startingHash
  );

  while (true){
    searcher.runGuess();
    std::cout << "Current processed: " << searcher.current_proccessed << std::endl;
  }

  return 0;
}
