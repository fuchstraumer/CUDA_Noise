#include "Perlin.h"
#include "..\cuda\cuda_assert.h"

namespace noise {

	namespace module {

		Perlin2D::Perlin2D(int width, int height, int seed) : Module(width, height) {

			// Setup perm table with unshuffled values.
			for (size_t c = 0; c < 255; ++c) {
				perm[c] = static_cast<unsigned char>(c);
			}

			// Shuffle permutation table.
			std::mt19937 rng;
			rng.seed(seed);
			std::shuffle(perm, perm + 256, rng);

			// Lookup arrays: 8 bits per channel, or a regular 32 bit-depth image/texture.

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

			// Malloc for arrays. Image format is read from cfDesc, and combined with the dimensions is used
			// to correct allocate.
			err = cudaMallocArray(&permArray, &cfDesc, 256, 256);
			cudaAssert(err);
			err = cudaMallocArray(&gradArray, &cfDesc, 256, 256);
			cudaAssert(err);

			// Copy to arrays. Can use "&vector[0]" to get pointer to vector's underling array, or just "vector.data()".
			err = cudaMemcpyToArray(permArray, 0, 0, &permutation[0], permutation.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaAssert(err);
			err = cudaMemcpyToArray(gradArray, 0, 0, &gradient[0], gradient.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaAssert(err);

			// Setup resource descriptors, which tie the actual resources (arrays) to CUDA objects
			// used in the kernels/device code (surfaces, textures)
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

			// Don't allow edge wrapping or looping, clamp to edges so out-of-range values
			// become edge values.
			permTDesc.addressMode[0] = cudaAddressModeWrap;
			permTDesc.addressMode[1] = cudaAddressModeWrap;
			permTDesc.addressMode[2] = cudaAddressModeWrap;
			gradTDesc.addressMode[0] = cudaAddressModeWrap;
			gradTDesc.addressMode[1] = cudaAddressModeWrap;
			gradTDesc.addressMode[2] = cudaAddressModeWrap;

			// No filtering, this is important to set. Otherwise our values we want to be exact will be linearly interpolated.
			permTDesc.filterMode = cudaFilterModePoint;
			gradTDesc.filterMode = cudaFilterModePoint;

			// Don't make the int data in this texture floating-point. Only counts for the CUDA-exclusive elements of the code.
			// Data is still 32 bits per pixel/element, and if we copy it back to the CPU there's nothing stopping us from treating
			// it like floating-point data.
			permTDesc.readMode = cudaReadModeElementType;
			gradTDesc.readMode = cudaReadModeElementType;

			// Normalized coords for permTDesc
			permTDesc.normalizedCoords = false;
			gradTDesc.normalizedCoords = false;

			// Create texture objects now
			permTex = 0;
			cudaAssert(cudaCreateTextureObject(&permTex, &permDesc, &permTDesc, nullptr));
			gradTex = 0;
			cudaAssert(cudaCreateTextureObject(&gradTex, &gradDesc, &gradTDesc, nullptr));

			// We pass the above textures into our FBM/Billow/Ridged/Swiss kernels and only need texture lookups now!
			// Cuts down size of device code and makes it easier to read, but also has HUGE speed benefits due to the 
			// cache-friendly nature of textures (both in access times AND cache locality)!
		}

		Perlin2D::~Perlin2D(){
		}

	}

}