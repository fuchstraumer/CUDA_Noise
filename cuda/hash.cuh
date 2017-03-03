#ifndef HASH_CUH
#define HASH_CUH
#include "common\CUDA_Include.h"
/*
	
	Hashing methods from accidental noise

	These have the tremendous benefit of letting us avoid
	LUTs!

*/
typedef unsigned int uint;
typedef unsigned char uchar;
// Hashing constants.
__device__ __constant__ uint FNV_32_PRIME = 0x01000193;
__device__ __constant__ uint FNV_32_INIT = 2166136261;
__device__ __constant__ uint FNV_MASK_8 = (1 << 8) - 1;

inline __device__ uint fnv_32_a_buf(const void* buf, const uint len) {
	uint hval = FNV_32_INIT;
	uint *bp = (uint*)buf;
	uint *be = bp + len;
	while (bp < be) {
		hval ^= (*bp++);
		hval *= FNV_32_PRIME;
	}
	return hval;
}

inline __device__ uchar xor_fold_hash(const uint hash) {
	return (uchar)((hash >> 8) ^ (hash & FNV_MASK_8));
}

inline __device__ uint hash_2d(const int x, const int y, const int seed) {
	uint d[3] = { (uint)x, (uint)y, (uint)seed };
	return xor_fold_hash(fnv_32_a_buf(d, 3));
}

inline __device__ uint hash_3d(const int x, const int y, const int z, const int seed) {
	uint d[4] = { (uint)x, (uint)y, (uint)z, (uint)seed };
	return xor_fold_hash(fnv_32_a_buf(d, 4));
}

inline __device__ uint hash_float_2d(const float x, const float y, const int seed) {
	uint d[3] = { (uint)x, (uint)y, (uint)seed };
	return xor_fold_hash(fnv_32_a_buf(d, sizeof(float) * 3 / sizeof(uint)));
}

inline __device__ uint hash_float_3d(const float x, const float y, const float z, const int seed) {
	uint d[4] = { (uint)x, (uint)y, (uint)z, (uint)seed };
	return xor_fold_hash(fnv_32_a_buf(d, sizeof(float) * 4 / sizeof(uint)));
}

#endif // !HASH_CUH
