#include "PerlinTex.h"
#include "../ext/include/lodepng/lodepng.h"
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

		PerlinTexBase::PerlinTexBase(int width, int height, float origin_x, float origin_y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height), Attributes(seed, freq, lacun, octaves, persist), Origin(origin_x, origin_y) {
			
			std::mt19937 rng;
			rng.seed(seed);
			std::shuffle(perm, perm + 256, rng);

			// create permutation table
			std::vector<unsigned char> perm_vec;
			perm_vec.resize(256 * 256 * 4);
			for (int j = 0; j < 256; ++j) {
				for (int i = 0; i < 256; ++i) {
					unsigned char a = perm[i] + static_cast<unsigned char>(j);
					unsigned char b = perm[(i + 1) & 255] + static_cast<unsigned char>(j);
					perm_vec[4 * 256 * j + 4 * i] = perm[a];
					perm_vec[4 * 256 * j + 4 * i + 1] = perm[(a + 1) & 255];
					perm_vec[4 * 256 * j + 4 * i + 2] = perm[b];
					perm_vec[4 * 256 * j + 4 * i + 3] = perm[(b + 1) & 255];
				}
			}
			lodepng::encode("permutation.png", &perm_vec[0], 256, 256);

			// create gradient table.
			/*std::vector<unsigned char> grad;
			grad.resize(256 * 256 * 4);
			uint32_t* Grad2DTable = new uint32_t[256 * 256];
			uint32_t* ptr = Grad2DTable;
			for (int y = 0; y < 256; ++y){
				for (int x = 0; x < 256; ++x){
					unsigned char px = perm[x];
					unsigned char py = perm[y];
					*ptr++ = (vector_table[((px & 7) << 1)] << 24) +
						(vector_table[((px & 7) << 1) + 1] << 16) +
						(vector_table[((py & 7) << 1)] << 8) +
						(vector_table[((py & 7) << 1) + 1]);
				}
			}*/
			std::vector<unsigned char> gradientv;
			gradientv.resize(256 * 256 * 4);
			for (int j = 0; j < 256; ++j) {
				for (int i = 0; i < 256; ++i) {
					unsigned char px = perm[i];
					unsigned char py = perm[j];
					gradientv[4 * 256 * j + 4 * i] = vector_table[(px & 7) << 1];
					gradientv[4 * 256 * j + 4 * i + 1] = vector_table[(px & 7) << 1];
					gradientv[4 * 256 * j + 4 * i + 2] = vector_table[(py & 7) << 1];
					gradientv[4 * 256 * j + 4 * i + 3] = vector_table[(py & 7) << 1];
				}
			}
			lodepng::encode("grad.png", &gradientv[0], 256, 256);

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
			err = cudaMemcpyToArray(permutationArray, 0, 0, &perm_vec[0], perm_vec.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaAssert(err);
			err = cudaMemcpyToArray(gradientArray, 0, 0, &gradientv[0], 256 * 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);
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
			permTDesc.addressMode[0] = cudaAddressModeClamp;
			permTDesc.addressMode[1] = cudaAddressModeClamp;
			permTDesc.addressMode[2] = cudaAddressModeClamp;
			gradTDesc.addressMode[0] = cudaAddressModeClamp;
			gradTDesc.addressMode[1] = cudaAddressModeClamp;
			gradTDesc.addressMode[2] = cudaAddressModeClamp;

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
			texFBMLauncher(output, permutation, gradient, dims.first, dims.second, make_float2(0.12312f, -12.312312f), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			cudaDeviceSynchronize();
			cudaError_t err = cudaSuccess;
			// Destroy LUTs
			//err = cudaDestroyTextureObject(permutation);
			//cudaAssert(err);
			//err = cudaDestroyTextureObject(gradient);
			//cudaAssert(err);
			//err = cudaFreeArray(permutationArray);
			//cudaAssert(err);
			//err = cudaFreeArray(gradientArray);
			//cudaAssert(err);
		}

		int PerlinTexBase::GetSourceModuleCount() const{
			return 0;
		}
	}
}
