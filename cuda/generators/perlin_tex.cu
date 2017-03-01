#include "perlin_tex.cuh"

__device__ float dot(const float x0, const float y0, const float x1, const float y1) {
	return (x0 * x1 + y0 * y1);
}

__device__ float dot(const float x0, const float y0, const float z0, const float w0, const float x1, const float y1, const float z1, const float w1) {
	return (x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1);
}

__device__ float perlin2d_tex(cudaTextureObject_t permutation, cudaTextureObject_t gradient, const float px, const float py, const int seed) {
	// Calculate 2D integer coordinates and fractional component 
	float2 i = make_float2(floorf(px), floorf(py));
	float2 f = make_float2(px - i.x, py - i.y);

	// Get weights.
	float4 w4 = make_float4(1.0f, 
		f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f),
		f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f), 
		(f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f)) * (f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f)));

	// Get four randomly permutated indices from the noise lattice nearest "point"
	// and offset them by the seed.
	uchar4 tmp = tex2D<uchar4>(permutation, i.x + 0.50f, i.y + 0.50f);
	float4 perm = make_float4(tmp.x + seed, tmp.y + seed, tmp.z + seed, tmp.w + seed);

	// Permute the fourst indices again and get the 2D gradient for each of
	// the four new coord-seed pairs.
	float4 gLeft, gRight;
	uchar4 tmp0 = tex2D<uchar4>(gradient, perm.x + 0.50f, perm.y + 0.50f);
	gLeft = make_float4(tmp0.x * 2.0f - 1.0f, tmp0.y * 2.0f - 1.0f, tmp0.z * 2.0f - 1.0f, tmp0.w * 2.0f - 1.0f);
	uchar4 tmp1 = tex2D<uchar4>(gradient, perm.z + 0.50f, perm.w + 0.50f);
	gRight = make_float4(tmp1.x * 2.0f - 1.0f, tmp1.y * 2.0f - 1.0f, tmp1.z * 2.0f - 1.0f, tmp1.w * 2.0f - 1.0f);

	// Evaluate gradients at four lattice points.
	float nLeftTop = dot(gLeft.x, gLeft.y, f.x, f.y);
	float nRightTop = dot(gRight.x, gRight.y, f.x - 1.0f, f.y);
	float nLeftBottom = dot(gLeft.z, gLeft.w, f.x, f.y - 1.0f);
	float nRightBottom = dot(gRight.z, gRight.w, f.x - 1.0f, f.y - 1.0f);

	// Blend gradients.
	float4 gradientBlend = make_float4(nLeftTop, nRightTop - nLeftTop, nLeftBottom - nLeftTop,
		nLeftTop - nRightTop - nLeftBottom + nRightBottom);
	float n = dot(nLeftTop, nRightTop - nLeftTop, nLeftBottom - nLeftTop, nLeftTop - nRightTop - nLeftBottom + nRightBottom,
		w4.x, w4.y, w4.z, w4.w);

	// Return value.
	return (n * 1.5f) / (2.5f);
}

__device__ float FBM2d_tex(cudaTextureObject_t permutation, cudaTextureObject_t gradient, float px, float py, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	float amplitude = 1.0f;
	// Scale point by freq
	px *= freq;
	py *= freq;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	float result = 0.0f;
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		result += perlin2d_tex(permutation, gradient, px, py, seed) * amplitude;
		// Modify vars for next octave.
		px *= lacun;
		py *= lacun;
		amplitude *= persist;
	}

	return result;
}

__global__ void texFBMKernel(cudaSurfaceObject_t output, cudaTextureObject_t permutation, cudaTextureObject_t gradient, const int width, const int height, const float2 origin, const float freq, const float lacun, const float persist, const int seed, const int octaves) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < width && j < height) {
		float2 p = make_float2(origin.x + i, origin.y + j);
		// Call billow function
		float val = FBM2d_tex(permutation, gradient, p.x, p.y, freq, lacun, persist, seed, octaves);
		// Write val to the surface
		surf2Dwrite(val, output, i * sizeof(float), j);
	}
}

void texFBMLauncher(cudaSurfaceObject_t output, cudaTextureObject_t permutation, cudaTextureObject_t gradient, const int width, const int height, const float2 origin, const float freq, const float lacun, const float persist, const int seed, const int octaves){
#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	texFBMKernel<<<numBlocks, threadsPerBlock>>>(output, permutation, gradient, width, height, origin, freq, lacun, persist, seed, octaves);
	// Check for succesfull kernel launch
	cudaAssert(cudaGetLastError());
	// Synchronize device
	cudaAssert(cudaDeviceSynchronize());

#ifdef CUDA_TIMING_TESTS
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_TIMING_TESTS
}