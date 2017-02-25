#ifndef PERLIN_CUH
#define PERLIN_CUH
#include "common\CUDA_Include.h"
#include "cuda_assert.h"
#include "vector_operators.cuh"
typedef unsigned char uchar;

/*

	perlin.cuh - CUDA header for base perlin generator functions

	Note that this noise is no longer octaved: this is not perlin noise.

	Octaved noise is a variant of perlin noise, and we shouldn't refer to perlin (in its raw form)
	anwwhere in this program, as its unlikely anyone would use raw noise. 

	Instead, this header and the respective .cu file will be used by most of the other generators
	as one of the base noise types.

*/

__device__ float perlin2d_tex(cudaTextureObject_t perm, cudaTextureObject_t grad, float2 point, int seed);

// From: https://github.com/covexp/cuda-noise/blob/master/cudaNoise/cudaNoise.cu
__device__ float perlin2d(float2 position, float scale, int seed);

// Actual noise function. Perlin2D just calls this with a randomly offset z coord.
__device__ float perlin3d(float3 position, float scale, int seed);



#endif // !PERLIN_2D_CUH
