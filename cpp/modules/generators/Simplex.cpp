#include "Simplex.h"
#include "../cuda/cuda_assert.h"

noise::module::Simplex2D::Simplex2D(int width, int height, int seed) : Module(width, height) {
	// Setup perm table with unshuffled values.
	for (size_t c = 0; c < 255; ++c) {
		perm[c] = static_cast<unsigned char>(c);
	}

	// Shuffle permutation table. (first, generate and seed an RNG instance)
	std::mt19937 rng;
	rng.seed(seed);
	std::shuffle(perm, perm + 256, rng);

	// Create channel format descriptions for our two lookup objects
	// perm-channel-format-Desc
	cudaChannelFormatDesc pcfDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	// gradient-channel-format-Desc
	cudaChannelFormatDesc gcfDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Allocate for arrays on device.
	cudaError_t err = cudaSuccess;
	err = cudaMallocArray(&pArray, &pcfDesc, 512, 1);
	cudaAssert(err);
	err = cudaMallocArray(&pArray, &pcfDesc, 8, 2);
	cudaAssert(err);

	// Copy data from host to device arrays
	err = cudaMemcpyToArray(pArray, 0, 0, &perm, 512 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaAssert(err);
	err = cudaMemcpyToArray(gArray, 0, 0, &gradientLUT, 8 * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaAssert(err);

	// Create resource descriptions, binding the above arrays to them
	struct cudaResourceDesc prDesc;
	struct cudaResourceDesc grDesc;
	memset(&prDesc, 0, sizeof(prDesc));
	memset(&grDesc, 0, sizeof(grDesc));

	// Set resource type
	prDesc.resType = cudaResourceTypeArray;
	grDesc.resType = cudaResourceTypeArray;

	// Bind array to resource description objects
	prDesc.res.array.array = pArray;
	grDesc.res.array.array = gArray;

	// Create texture descriptions.
	struct cudaTextureDesc ptDesc;
	struct cudaTextureDesc gtDesc;
	memset(&ptDesc, 0, sizeof(ptDesc));
	memset(&gtDesc, 0, sizeof(gtDesc));



}