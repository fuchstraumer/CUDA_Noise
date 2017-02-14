#include "stdafx.h"
#include "Perlin.h"
#include "..\cuda\generators\perlin.cuh"
#include "..\cuda\cuda_assert.h"
namespace noise {
	
	namespace module {

		// Pass width and height to base class ctor, initialize configuration struct, initialize origin (using initializer list)
		Perlin2D::Perlin2D(int width, int height, int x, int y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height), 
			Attributes(freq, lacun, persist, octaves, seed, PERLIN_MAX_OCTAVES), Origin(x,y) {
			
			// Setup lookup texture object.

			// Permutation table.
			unsigned char perm[512];
			
			// Setup perm table with unshuffled values.
			for (size_t c = 0; c < 255; ++c) {
				perm[c] = static_cast<unsigned char>(c);
				perm[c + 256] = static_cast<unsigned char>(c);
			}

			// Shuffle permutation table.
			std::shuffle(perm, perm + 512, std::default_random_engine());

			// C++ work done and our objects are good to go. Now, create CUDA resource and texture descriptions
			// telling CUDA what this data is, how to read this data, and allocate for it + create the objects
			// so we can pass them to the kernel when the time comes.

			// Setup first array, for the permutation table. This will be a 1D texture.

			// Channel format: only one channel, X, of type 8-bit unsigned (unsigned char)
			cudaChannelFormatDesc permcfDescr = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

			// Create array and allocate space for it on the device. (which is why we need permDescr)
			cudaArray* permArray;
			cudaMallocArray(&permArray, &permcfDescr, width, height);

			// Copy data from perm -> permArray 
			cudaMemcpyToArray(permArray, 0, 0, &perm, sizeof(perm), cudaMemcpyHostToDevice);

			// Now begin specifying resource params.
			struct cudaResourceDesc permRDescr;
			memset(&permRDescr, 0, sizeof(permRDescr));

			// Specify type of data, in this case a CUDA Array
			permRDescr.resType = cudaResourceTypeArray;

			// Bind the description to the CUDA array we create above
			permRDescr.res.array.array = permArray;

			// Specify texture data and params
			struct cudaTextureDesc permTDescr;
			memset(&permTDescr, 0, sizeof(permTDescr));
			permTDescr.readMode = cudaReadModeElementType;

			// Lastly, create the texture object for Perm.
			permutation = 0;
			cudaCreateTextureObject(&permutation, &permRDescr, &permTDescr, nullptr);

		}

		// TODO: Implement these. Just here so compiler shuts up.
		int Perlin2D::GetSourceModuleCount() const{
			return 0;
		}

		void Perlin2D::Generate(){
			PerlinLauncher(output, permutation, dims.x, dims.y, make_float2(Origin.x, Origin.y), Attributes.Frequency,
				Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
		}
	}
}
