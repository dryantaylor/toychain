#include "SearchControl.hpp"

SearchControl::SearchControl(std::unique_ptr<BulkHasher> hasher, std::ostream& outstream,
                             const std::string& name, SHA256Hash startingHash ) :
                             hasher{std::move(hasher)},
                             outstream(outstream),
                             name(name),
                             currentHash(startingHash),
                             randomGen(std::random_device{}()),
                             distrib(0, UINT64_MAX)
{
}

void SearchControl::runGuess()
{
  std::vector<std::string> plainTexts;
  for (std::size_t i = 0; i < hasher->getBatchSize(); i++){

    std::uint64_t nonce = distrib(randomGen);
    std::string nonce_str = std::to_string(nonce);

    std::string plainText = currentHash.to_string() + name + nonce_str;
    plainTexts.push_back(plainText);
  }

  hasher->setHashes(plainTexts);
  hasher->computeHash();
  hasher->computeIsMoreDifficult();
  std::vector<bool> difficulty = hasher->getIsMoreDifficult();
  std::vector<SHA256Hash> hashes = hasher->getHashes();

  bool moreDifficultFound = false;
  int maxDifficulty = 0;
  std::size_t hashIndex = 0;

  for (std::size_t i = 0; i < hasher->getBatchSize(); i++){
    if (difficulty[i] && hashes.at(i).difficulty() > maxDifficulty){
      moreDifficultFound = true;
      hashIndex = i;
      maxDifficulty = hashes.at(i).difficulty();
      
    }
  }

  if (moreDifficultFound){
    hasher->setNewDifficulty(hashes.at(hashIndex).difficulty() + 1);
    currentHash = hashes.at(hashIndex);
    outstream << plainTexts.at(hashIndex) << std::endl;
    std::cout << "Soloution Found -" << plainTexts.at(hashIndex) << "   -"<< currentHash.difficulty()<< std::endl;
  }
}

