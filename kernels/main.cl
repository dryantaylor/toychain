__kernel void compute(__global const uchar* input_blob,
                           __global const int* input_lengths,
                           __global const int* input_offsets,
                           __global ulong* output_hashes,
                           __global bool* output_difficulty,
                           const int targetDifficulty,
                           const int count) {
  generate_hash(input_blob, input_lengths, input_offsets, output_hashes, count);
  calculateMeetsDifficulty(output_hashes, output_difficulty,targetDifficulty);
  }