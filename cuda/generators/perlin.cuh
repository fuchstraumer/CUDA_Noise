#ifndef FBM_CUH
#define FBM_CUH
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

#ifndef HALF_PRECISION_SUPPORT

// 32-bit device functions for generating FBM noise.

__device__ float perlin2d(cudaTextureObject_t perm, cudaTextureObject_t grad, float2 point, int seed);

#endif // !HALF_PRECISION_SUPPORT

#ifdef HALF_PRECISION_SUPPORT

#endif // !HALF_PRECISION_SUPPORT


#endif // !PERLIN_2D_CUH
