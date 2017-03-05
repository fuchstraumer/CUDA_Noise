#ifndef FBM_CUH
#define FBM_CUH
#include "../common/CUDA_Include.h"
#include "../noise_generators.cuh"

__device__ float FBM2d_Simplex(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves);

__device__ float FBM2d(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves);

void FBM_Launcher(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !FBM_CUH
