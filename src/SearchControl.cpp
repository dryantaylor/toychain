#include "SearchControl.hpp"

#include <iostream>

SearchControl::SearchControl(std::unique_ptr<BulkHasher> hasher, std::ostream& outstream,
                             const std::string& name, SHA256Hash startingHash ) :
                             hasher{std::move(hasher)},
                             outstream(outstream),
                             name(name),
                             currentHash(startingHash),
                             randomGen(std::random_device{}()),
                             distrib(0, UINT64_MAX)
{
  std::string prefix = startingHash.to_string()+name;
  this->hasher->setString(prefix);
}

void SearchControl::runGuess()
{
  hasher->compute();

  std::vector<bool> difficulty = hasher->getIsMoreDifficult();
  std::vector<SHA256Hash> hashes = hasher->getHashes();
  bool moreDifficultFound = false;
  int maxDifficulty = 0;
  std::size_t hashIndex = 0;
  for (std::size_t i = 0; i < difficulty.size(); i++) {
    if (difficulty[i] && hashes.at(i).difficulty() > maxDifficulty) {
      moreDifficultFound = true;
      hashIndex = i;
      maxDifficulty = hashes.at(i).difficulty();
    }
  }

  if (moreDifficultFound) {
    hasher->setString(hashes.at(hashIndex).to_string() + name);
    hasher->setNewDifficulty(hashes.at(hashIndex).difficulty() + 1);
    outstream << hasher->getPlaintextAt(hashIndex) << std::endl;
    std::cout << "New Hash found: " << hasher->getPlaintextAt(hashIndex)
    << " - " << hashes.at(hashIndex).difficulty() << std::endl;
  }
  current_proccessed += hasher->getBatchSize();
}

