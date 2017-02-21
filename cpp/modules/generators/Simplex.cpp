#include "Simplex.h"
#include "../cuda/cuda_assert.h"

noise::module::Simplex2D::Simplex2D(int width, int height, int seed) : Module(width, height) {
	// Permutation lookup table that is generated
	// and then passed to the CUDA API as a CUDA Array
	std::vector<unsigned char> perm;
	perm.resize(512);

	// Gradient vector lookup table, also passed to the
	// API.
	std::vector<float> gradientLUT = {
		 -1.0f,-1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f,1.0f,
		 -1.0f, 1.0f, 0.0f,-1.0f, 0.0f, 1.0f, 1.0f,-1.0f ,
	};
	
	// Setup perm table with unshuffled values.
	for (size_t c = 0; c < 256; ++c) {
		perm[c] = static_cast<unsigned char>(c);
		perm[c + 255] = static_cast<unsigned char>(c);
	}

	// Shuffle permutation table. (first, generate and seed an RNG instance)
	std::mt19937 rng;
	rng.seed(seed);
	std::shuffle(perm.begin(), perm.end(), rng);

	// Create channel format descriptions for our two lookup objects
	// perm-channel-format-Desc
	cudaChannelFormatDesc pcfDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	// gradient-channel-format-Desc
	cudaChannelFormatDesc gcfDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Allocate for arrays on device.
	cudaError_t err = cudaSuccess;
	err = cudaMallocArray(&pArray, &pcfDesc, 512, 0);
	cudaAssert(err);
	err = cudaMallocArray(&gArray, &gcfDesc, 8, 2);
	cudaAssert(err);

	// Copy data from host to device arrays
	err = cudaMemcpyToArray(pArray, 0, 0, &perm[0], sizeof(unsigned char) * perm.size(), cudaMemcpyHostToDevice);
	cudaAssert(err);
	err = cudaMemcpyToArray(gArray, 0, 0, &gradientLUT[0], sizeof(float) * gradientLUT.size(), cudaMemcpyHostToDevice);
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

	// Specify read type, filtering, border/wrapping

	// Don't allow edge wrapping or looping, clamp to edges so out-of-range values
	// become edge values.
	ptDesc.addressMode[0] = cudaAddressModeWrap;
	ptDesc.addressMode[1] = cudaAddressModeWrap;
	ptDesc.addressMode[2] = cudaAddressModeWrap;
	gtDesc.addressMode[0] = cudaAddressModeWrap;
	gtDesc.addressMode[1] = cudaAddressModeWrap;
	gtDesc.addressMode[2] = cudaAddressModeWrap;

	// No filtering, this is important to set. Otherwise our values we want to be exact will be linearly interpolated.
	ptDesc.filterMode = cudaFilterModePoint;
	gtDesc.filterMode = cudaFilterModePoint;

	// Don't make the int data in this texture floating-point. Only counts for the CUDA-exclusive elements of the code.
	// Data is still 32 bits per pixel/element, and if we copy it back to the CPU there's nothing stopping us from treating
	// it like floating-point data.
	ptDesc.readMode = cudaReadModeElementType;
	gtDesc.readMode = cudaReadModeElementType;

	// Normalized coords for permTDesc
	ptDesc.normalizedCoords = false;
	gtDesc.normalizedCoords = false;

	// Create texture objects now
	pTex = 0;
	cudaAssert(cudaCreateTextureObject(&pTex, &prDesc, &ptDesc, nullptr));
	gTex = 0;
	cudaAssert(cudaCreateTextureObject(&gTex, &grDesc, &gtDesc, nullptr));
}