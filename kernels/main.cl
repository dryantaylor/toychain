static inline ulong rotl64(ulong value, int bits) {
  return (value << bits) | (value >> (64 - bits));
}

static inline ulong2 longlongrnd(ulong2 state, int gid)
{
  ulong2 s = state + (ulong2)(gid, gid + 0x9E3779B97F4A7C15UL);

  ulong lo = s.x;
  ulong hi = s.y;

  if ((lo | hi) == 0UL) {
    lo = 0xD1B54A35C4E1F6ADUL;
    hi = 0x94D049BB133111EBUL;
  }

  const ulong x = lo + hi;
  lo ^= rotl64(lo ^ hi, 25U);
  hi = rotl64(hi ^ x, 17U) ^ lo;
  lo = rotl64(lo, 31U);

  return (ulong2)(lo, hi);
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

__kernel void compute(ulong seed_lo,
                      ulong seed_hi,
                      __global const char* string_prefix,
                      int string_size,
                      __global long* winnerNonce,
                      __global ulong* winnerHashes,
                      __global int* matchCount,
                      int targetDifficulty,
                      int batchSize) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int localSize = get_local_size(0);

  if (gid >= batchSize) {
    return;
  }

  __local int localMatchCount;
  __local long localWinnerNonce[256];
  __local ulong localWinnerHashes[256 * 4];

  if (lid == 0) {
    localMatchCount = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  ulong2 state = (ulong2)(seed_lo, seed_hi);
  ulong2 rnd = longlongrnd(state, gid);
  long nonce = convert_long(rnd.x ^ rotl64(rnd.y, 13U));

  char candidate[256];
  for (int i = 0; i < string_size; ++i) {
    candidate[i] = string_prefix[i];
  }

  int final_size = 0;
  ulong hash[4];

  long_to_str(nonce, candidate, string_size, &final_size);
  sha256_hash_string((__private uchar*)candidate, final_size, hash);

  if (hashMeetsDifficultyPrivate(hash, targetDifficulty)) {
    int slot = atomic_add(&localMatchCount, 1);
    if (slot < 256) {
      localWinnerNonce[slot] = nonce;
      localWinnerHashes[slot * 4 + 0] = hash[0];
      localWinnerHashes[slot * 4 + 1] = hash[1];
      localWinnerHashes[slot * 4 + 2] = hash[2];
      localWinnerHashes[slot * 4 + 3] = hash[3];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    int base = atomic_add(matchCount, localMatchCount);
    for (int i = 0; i < localMatchCount; ++i) {
      winnerNonce[base + i] = localWinnerNonce[i];
      winnerHashes[(base + i) * 4 + 0] = localWinnerHashes[i * 4 + 0];
      winnerHashes[(base + i) * 4 + 1] = localWinnerHashes[i * 4 + 1];
      winnerHashes[(base + i) * 4 + 2] = localWinnerHashes[i * 4 + 2];
      winnerHashes[(base + i) * 4 + 3] = localWinnerHashes[i * 4 + 3];
    }
  }
}