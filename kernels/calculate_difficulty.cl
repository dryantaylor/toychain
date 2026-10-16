static bool hashMeetsDifficulty(const __global ulong* hash, int difficulty) {
  if (difficulty <= 0) {
    return true;
  }
  if (difficulty > 256) {
    return false;
  }

  const int shift = 256 - difficulty;
  const int wordIndex = 3 - (shift / 64);
  const int bitIndex = shift % 64;

  for (int i = 0; i < wordIndex; ++i) {
    if (hash[i] != 0UL) {
      return false;
    }
  }

  const ulong wordThreshold = 1UL << bitIndex;
  return hash[wordIndex] < wordThreshold;
}

static void calculateMeetsDifficulty(__global const ulong* hashes, __global bool* output, int difficulty){
  int gid = get_global_id(0);
  output[gid] = hashMeetsDifficulty(&hashes[gid*4], difficulty);
}