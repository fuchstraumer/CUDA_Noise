#ifndef SELECT_CUH
#define SELECT_CUH
#include "common\CUDA_Include.h"

__global__ void SelectKernel(cudaSurfaceObject_t out, cudaTextureObject_t select_item, cudaTextureObject_t subject0, cudaTextureObject_t subject1, int width, int height, float upper_bound, float lower_bound, float falloff);

void SelectLauncher(cudaSurfaceObject_t out, cudaTextureObject_t select_item, cudaTextureObject_t subject0, cudaTextureObject_t subject1, int width, int height, float upper_bound, float lower_bound, float falloff);

#endif // !SELECT_CUH
