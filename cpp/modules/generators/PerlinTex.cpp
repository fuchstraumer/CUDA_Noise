#include "PerlinTex.h"
#include "../cuda/generators/perlin_tex.cuh"
namespace noise {

	namespace module {

		static unsigned char perm[256] = { 151,160,137,91,90,15,
			131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
			190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
			88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
			77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
			102,143,54,65,25,63,161,255,216,80,73,209,76,132,187,208,89,18,169,200,196,
			135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
			5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
			223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
			129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
			251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
			49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
			138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };
		// Setup texture LUT
		static uint8_t vector_table[16] = {
			245, 176,
			176, 245,
			79, 245,
			10, 176,
			10, 79,
			79, 10,
			176, 10,
			245, 79
		};

		PerlinTexBase::PerlinTexBase(int width, int height, float2 origin, int seed) : Module(width, height){
			
			std::mt19937 rng;
			rng.seed(seed);
			std::shuffle(perm, perm + 256, rng);

			// create permutation table
			uint32_t* perm_table = new uint32_t[256 * 256];
			uint32_t* ptr = perm_table;
			for (int y = 0; y < 256; ++y) {
				for (int x = 0; x < 256; ++x) {
					unsigned char a = perm[x] + static_cast<unsigned char>(y);
					unsigned char b = perm[(x + 1) & 255] + static_cast<unsigned char>(x);
					*ptr++ = (perm[a] << 24) + (perm[(a + 1) & 255] << 16) + (perm[b] << 8) + (perm[(b + 1) & 255]);
				}
			}

			// create gradient table.
			uint32_t* grad_table = new uint32_t[256 * 256];
			ptr = grad_table;
			for (int y = 0; y < 256; ++y) {
				for (int x = 0; x < 256; ++x) {
					unsigned char px = perm[x];
					unsigned char py = perm[y];
					*ptr++ = (vector_table[((px & 7) << 1)] << 24) + (vector_table[((px & 7) << 1) + 1] << 16) + (vector_table[(py & 7) << 1] << 8) + (vector_table[((py & 7) << 1) + 1]);
				}
			}

			// With these objects setup, time to pass them to the GPU/CUDA
			// Channel format description: (8,8,8,8), unsigned char
			cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

			cudaError_t err = cudaSuccess;

			// Malloc for arrays. Image format is read from cfDesc, and combined with the dimensions is used
			// to correct allocate.
			err = cudaMallocArray(&permutationArray, &cfDesc, 256, 256);
			cudaAssert(err);
			err = cudaMallocArray(&gradientArray, &cfDesc, 256, 256);
			cudaAssert(err);

			// Copy to arrays. Can use "&vector[0]" to get pointer to vector's underling array, or just "vector.data()".
			err = cudaMemcpyToArray(permutationArray, 0, 0, perm_table, (256 * 256) * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaAssert(err);
			err = cudaMemcpyToArray(gradientArray, 0, 0, grad_table, (256 * 256) * sizeof(unsigned char), cudaMemcpyHostToDevice);
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
			permDesc.res.array.array = permutationArray;
			gradDesc.res.array.array = gradientArray;

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
			permutation = 0;
			cudaAssert(cudaCreateTextureObject(&permutation, &permDesc, &permTDesc, nullptr));
			gradient = 0;
			cudaAssert(cudaCreateTextureObject(&gradient, &gradDesc, &gradTDesc, nullptr));
		}

		void PerlinTexBase::Generate(){
			texFBMLauncher(output, permutation, gradient, dims.first, dims.second, make_float2(0.0f, 0.0f), 1.0f, 1.6f, 0.9f, 22134123, 12);
			cudaDeviceSynchronize();
			cudaError_t err = cudaSuccess;
			// Destroy LUTs
			err = cudaDestroyTextureObject(permutation);
			cudaAssert(err);
			err = cudaDestroyTextureObject(gradient);
			cudaAssert(err);
			err = cudaFreeArray(permutationArray);
			cudaAssert(err);
			err = cudaFreeArray(gradientArray);
			cudaAssert(err);
		}

		int PerlinTexBase::GetSourceModuleCount() const{
			return 0;
		}
	}
}
