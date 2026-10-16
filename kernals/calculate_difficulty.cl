static bool hashMeetsDifficulty( const __global ulong* hash, int difficulty){

  const int shift = 256 - difficulty;

  const size_t wordIndex = 3 - ( shift / 64); //word with the final required leading zero (words are 0 index on MSW, 3 on LSW)
  const size_t bitIndex = (shift % 64);  //bit index of the final required leading 0 (0 index on LSB, 64 index on MSB)

  // any words greater than the target word should be 0.
  // i.e. if difficulty is 65 then the first 64 bits (0th word) should be 0
  //for (size_t i = 0; i < wordIndex; i++){
  //  if (hash[i] > 0) {
  //    return false;
  //  }
  //}

  //anything less than this threshold will be 0
  //i.e if difficulty check is >=4 than it must start with atleast 0000....
  //therefor anything less than 0001..... will have atleast 4 leading 0's
  ulong wordThreshold = 0b1UL << bitIndex;

  bool ok  = hash[0] <= (wordIndex == 0 ? wordThreshold-1 : (0 < wordIndex ? 0 : ULONG_MAX));
       ok &= hash[1] <= (wordIndex == 1 ? wordThreshold-1 : (1 < wordIndex ? 0 : ULONG_MAX));
       ok &= hash[2] <= (wordIndex == 2 ? wordThreshold-1 : (2 < wordIndex ? 0 : ULONG_MAX));
       ok &= hash[3] <= (wordIndex == 3 ? wordThreshold-1 : (3 < wordIndex ? 0 : ULONG_MAX));
  //return  hash[wordIndex] < wordThreshold;
  return ok;
  // the words less significant than our wordIndex don't matter

}

__kernel void calculateMeetsDifficulty(__global const ulong* hashes, __global bool* output, int difficulty){
  int gid = get_global_id(0);
  output[gid] = hashMeetsDifficulty(&hashes[gid*4], difficulty);
}