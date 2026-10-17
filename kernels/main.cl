static int intrnd (int seed) // 1<=seed<m
{
  int const a = 16807; //ie 7**5
  int const m = 2147483647; //ie 2**31-1
  seed = (convert_long(seed * a)) %m;
  return seed;
}


static void long_to_str(long value, char* output_str,int startOffset, int* output_size){
  uchar digits[20];

  int count = 0;
  
  digits[count++] = (uchar)('0' + value%10U);
  value /=10U;
  while (value > 0){
    digits[count++] = (uchar)('0' + value%10U);
    value/=10u;
  }
  *output_size = 0;
  // reverse digits into the output buffer
    for (int i = count - 1; i >= 0; --i) {
        output_str[startOffset+*output_size] = digits[i];
        (*output_size)++;
    }
    output_str[startOffset+*output_size] = '\0';
    *output_size+=startOffset;
}

__kernel void compute(int seed1,
                      __global const char* string_prefix,
                      int string_size,
                      __global long* winnerNonce,
                      __global ulong* winnerHashes,
                      __global int* matchCount,
                      int targetDifficulty,
                      int batchSize) {
  int gid = get_global_id(0);
  if (gid >= batchSize) {
    return;
  }

  long nonce = convert_long(intrnd(seed1 + gid));

  char candidate[256];
  for (int i = 0; i < string_size; ++i) {
    candidate[i] = string_prefix[i];
  }

  int final_size = 0;
  ulong hash[4];

  long_to_str(nonce, candidate, string_size, &final_size);
  sha256_hash_string((__private uchar*)candidate, final_size, hash);

  if (hashMeetsDifficultyPrivate(hash, targetDifficulty)) {
    int slot = atomic_add(matchCount, 1);
    winnerNonce[slot] = nonce;
    winnerHashes[slot * 4 + 0] = hash[0];
    winnerHashes[slot * 4 + 1] = hash[1];
    winnerHashes[slot * 4 + 2] = hash[2];
    winnerHashes[slot * 4 + 3] = hash[3];
  }
}