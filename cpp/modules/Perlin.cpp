#include "stdafx.h"
#include "Perlin.h"
#include "..\cuda\cuda_assert.h"
namespace noise::module {

	Perlin2D::Perlin2D(int width, int height) : Module(width, height) {

		// Setup perm table with unshuffled values.
		for (size_t c = 0; c < 255; ++c) {
			perm[c] = static_cast<unsigned char>(c);
		}

		// Shuffle permutation table.
		std::shuffle(perm, perm + 256, std::default_random_engine());

		// Lookup arrays: 8 bits per channel, or 32 bits per single value
		

		// I'm using new blocks here to define generating these textures, mostly because
		// (in my opinion) it makes things seem a bit more organized and "clean"

		// Generate permutation texture data.
		std::vector<unsigned char> permutation;
		permutation.resize(256 * 256 * 4);
		for (int i = 0; i < 256; ++i) {
			for (int j = 0; j < 256; ++j) {
				unsigned char a = perm[i] + static_cast<unsigned char>(j);
				unsigned char b = perm[(i + 1) & 255] + static_cast<unsigned char>(j);
				permutation[4 * 256 * j + 4 * i] = perm[a];
				permutation[4 * 256 * j + 4 * i + 1] = perm[(a + 1) & 255];
				permutation[4 * 256 * j + 4 * i + 2] = perm[b];
				permutation[4 * 256 * j + 4 * i + 3] = perm[(b + 1) & 255];
			}
		}
	
		// Generate gradient texture data.
		// No need to generate these like we did with "perm": these are just lookups
		// for gradient vector angles
		static constexpr unsigned char grad[16]{
			245, 176,
			176, 245,
			79, 245,
			10, 176,
			10, 79,
			79, 10,
			176, 10,
			245, 79
		};

		std::vector<unsigned char> gradient;
		gradient.resize(256 * 256 * 4);
		for (int i = 0; i < 256; ++i) {
			for (int j = 0; j < 256; ++j) {
				unsigned char px = perm[i];
				unsigned char py = perm[j];
				gradient[4 * 256 * j + 4 * i] = grad[(px & 7) << 1];
				gradient[4 * 256 * j + 4 * i + 1] = grad[(px & 7) << 1];
				gradient[4 * 256 * j + 4 * i + 2] = grad[(py & 7) << 1];
				gradient[4 * 256 * j + 4 * i + 3] = grad[(py & 7) << 1];
			}
		}

		// Channel format description: (8,8,8,8), unsigned char
		cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

		cudaError_t err = cudaSuccess;

		// Malloc for arrays
		err = cudaMallocArray(&permArray, &cfDesc, 256, 256);
		cudaAssert(err);
		err = cudaMallocArray(&gradArray, &cfDesc, 256, 256);
		cudaAssert(err);

		// Copy to arrays
		err = cudaMemcpyToArray(permArray, 0, 0, &permutation[0], sizeof(permutation), cudaMemcpyHostToDevice);
		cudaAssert(err);
		err = cudaMemcpyToArray(permArray, 0, 0, &gradient[0], sizeof(gradient), cudaMemcpyHostToDevice);
		cudaAssert(err);

		// Setup resource descriptors
		struct cudaResourceDesc permDesc;
		struct cudaResourceDesc gradDesc;
		memset(&permDesc, 0, sizeof(permDesc));
		memset(&gradDesc, 0, sizeof(gradDesc));

		// Set resource type (array)
		permDesc.resType = cudaResourceTypeArray;
		gradDesc.resType = cudaResourceTypeArray;

		// Bind arrays to resource descriptions.
		permDesc.res.array.array = permArray;
		gradDesc.res.array.array = gradArray;

		// Setup texture descriptors
		struct cudaTextureDesc permTDesc;
		struct cudaTextureDesc gradTDesc;
		memset(&permTDesc, 0, sizeof(permTDesc));
		memset(&gradTDesc, 0, sizeof(gradTDesc));

		// Specify read type, filtering, border/wrapping
		permTDesc.addressMode[0] = cudaAddressModeClamp;
		permTDesc.addressMode[1] = cudaAddressModeClamp;
		permTDesc.addressMode[2] = cudaAddressModeClamp;
		gradTDesc.addressMode[0] = cudaAddressModeClamp;
		gradTDesc.addressMode[1] = cudaAddressModeClamp;
		gradTDesc.addressMode[2] = cudaAddressModeClamp;
		// No filtering, this is important to set.
		permTDesc.filterMode = cudaFilterModePoint;
		gradTDesc.filterMode = cudaFilterModePoint;
		// Don't make the int data in this texture floating-point
		permTDesc.readMode = cudaReadModeElementType;
		gradTDesc.readMode = cudaReadModeElementType;

		// Create texture objects now
		permTex = 0;
		cudaAssert(cudaCreateTextureObject(&permTex, &permDesc, &permTDesc, nullptr));
		gradTex = 0;
		cudaAssert(cudaCreateTextureObject(&gradTex, &gradDesc, &gradTDesc, nullptr));

		// We pass the above textures into our FBM/Billow/Ridged/Swiss kernels and only need texture lookups now!
	}

}