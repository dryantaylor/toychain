#ifndef SEARCH_CONTROL_HPP
#define SEARCH_CONTROL_HPP

#include "BulkHasher.hpp"
#include <memory>
#include <ostream>
#include "SHA256Hash.hpp"
#include <random>

class SearchControl {
  public:
    SearchControl(std::unique_ptr<BulkHasher> hasher, std::ostream& outstream,const std::string& name, SHA256Hash startingHash );
    void runGuess();
  
    std::uint64_t current_proccessed = 0;
  private:
  std::unique_ptr<BulkHasher> hasher;
  std::ostream& outstream;
  std::string name;
  SHA256Hash currentHash;
  std::mt19937 randomGen;
  std::uniform_int_distribution<std::uint64_t> distrib;

};

#endif