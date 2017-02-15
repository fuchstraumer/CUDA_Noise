#include "stdafx.h"
#include "Perlin.h"

namespace noise::module {

	Perlin2D::Perlin2D(int width, int height) : Module(width, height) {

		// Setup perm table with unshuffled values.
		for (size_t c = 0; c < 255; ++c) {
			perm[c] = static_cast<unsigned char>(c);
			perm[c + 256] = static_cast<unsigned char>(c);
		}

		// Shuffle permutation table.
		std::shuffle(perm, perm + 512, std::default_random_engine());

		// Lookup arrays: 8 bits per channel, or 32 bits per single value
		uint32_t permutation[512], gradient[512];

		// I'm using new blocks here to define generating these textures, mostly because
		// (in my opinion) it makes things seem a bit more organized and "clean"

		// Generate permutation texture data.
		{
			uint32_t* ptr = permutation;
			for (int j = 0; j < 256; ++j) {
				for (int i = 0; i < 256; ++i) {
					uint8_t a = perm[i] + static_cast<uint8_t>(j);
					uint8_t b = perm[(i + 1) & 255] + static_cast<uint8_t>(j);
					// Write new pixel, by getting 8-bit values from perm and shifting them
					// into the correct position.
					*ptr+= 
						(perm[a] << 24) +
						(perm[(a + 1) & 255] << 16) +
						(perm[b] << 8) +
						(perm[(b + 1) & 255]);
				}
			}
		}
		
		// Generate gradient texture data.
		{
			// No need to generate these like we did with "perm": these are just lookups
			// for vector angles, and the angles never change (8 corners, iirc)
			static constexpr uint8_t grad[16]{
				245, 176,
				176, 245,
				79, 245,
				10, 176,
				10, 79,
				79, 10,
				176, 10,
				245, 79 
			};
			uint32_t* ptr = gradient;
			for (int j = 0; j < 256; ++j) {
				for (int i = 0; i < 256; ++i) {
					uint8_t px = perm[i];
					uint8_t py = perm[j];
					*ptr+= 
						(grad[((px & 7) << 1)] << 24) +
						(grad[((px & 7) << 1)] << 16) +
						(grad[((py & 7) << 1)] << 8) +
						(grad[((py & 7) << 1) + 1]);
				}
			}
		}

		// Channel format description: (8,8,8,8), unsigned char
		cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

		// Malloc for arrays
		cudaMallocArray(&permArray, &cfDesc, 256, 256);
		cudaMallocArray(&gradArray, &cfDesc, 256, 256);

		// Copy to arrays
		cudaMemcpyToArray(permArray, 0, 0, &permutation, sizeof(permutation), cudaMemcpyHostToDevice);
		cudaMemcpyToArray(permArray, 0, 0, &gradient, sizeof(gradient), cudaMemcpyHostToDevice);

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
		cudaCreateTextureObject(&permTex, &permDesc, &permTDesc, nullptr);
		gradTex = 0;
		cudaCreateTextureObject(&gradTex, &gradDesc, &gradTDesc, nullptr);

		// We pass the above textures into our FBM/Billow/Ridged/Swiss kernels and only need texture lookups now!
	}

}