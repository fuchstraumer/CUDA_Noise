#ifndef SIMPLEX_CUH
#define SIMPLEX_CUH
#include "common\CUDA_Include.h"

__device__ float simplex2d(cudaTextureObject_t perm, cudaTextureObject_t grad, float2 point, int seed);

#endif // !SIMPLEX_3D_CUH
