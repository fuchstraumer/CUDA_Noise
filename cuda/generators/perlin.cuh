#ifndef PERLIN_CUH
#define PERLIN_CUH
#include "common\CUDA_Include.h"
#include "cuda_assert.h"
typedef unsigned char uchar;

/*

	perlin.cuh - CUDA header for base perlin generator functions

	Note that this noise is no longer octaved: this is not perlin noise.

	Octaved noise is a variant of perlin noise, and we shouldn't refer to perlin (in its raw form)
	anwwhere in this program, as its unlikely anyone would use raw noise. 

	Instead, this header and the respective .cu file will be used by most of the other generators
	as one of the base noise types.

*/

// From: https://github.com/covexp/cuda-noise/blob/master/cudaNoise/cudaNoise.cu
__device__ float perlin2d(const float2 position, const int seed);

// Actual noise function. Perlin2D just calls this with a randomly offset z coord.
__device__ float perlin3d(const float px, const float py, const float pz, const int seed, const noise_quality qual);

__device__ float perlin2d_dx(const float2 position, const int seed);

__device__ float perlin2d_dy(const float2 position, const int seed);

#endif // !PERLIN_2D_CUH
