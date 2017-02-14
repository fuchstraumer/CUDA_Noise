#pragma once
#ifndef FBM_H
#define FBM_H
#include "Perlin.h"
/*

	Generates values using FBM noise. Ctor includes desired dimensions
	of output texture.
	
	These classes must exist as wrappers over the CUDA kernels since we need
	to allocate and create our constant texture objects, used as permutation
	tables and the like. Without these, we have to use costly/slow array lookups.

*/
namespace noise {

	namespace module {

		// Default parameters
		constexpr float DEFAULT_FBM_FREQUENCY = 1.0f;
		constexpr float DEFAULT_FBM_LACUNARITY = 2.0f;
		constexpr int DEFAULT_FBM_OCTAVES = 6;
		constexpr float DEFAULT_FBM_PERSISTENCE = 0.50f;
		constexpr int DEFAULT_FBM_SEED = 0;

		// Maximum octave level to allow
		constexpr int FBM_MAX_OCTAVES = 24;


		class FBM2D : public Perlin2D {
		public:

			// Width + height specify output texture size.
			// Seed defines a value to seed the generator with
			// X & Y define the origin of the noise generator
			FBM2D(int width, int height, int x = 0, int y = 0, int seed = DEFAULT_FBM_SEED, float freq = DEFAULT_FBM_FREQUENCY, float lacun = DEFAULT_FBM_LACUNARITY,
				int octaves = DEFAULT_FBM_OCTAVES, float persist = DEFAULT_FBM_PERSISTENCE);

			// Get source module count: must be 0, this is a generator and can't have preceding modules.
			virtual int GetSourceModuleCount() const override;

			// Launches the kernel and fills this object's surface object with the relevant data.
			virtual void Generate() override;

			// Origin of this noise generator. Keep the seed constant and change this for 
			// continuous "tileable" noise
			glm::vec2 Origin;

			// Configuration attributes.
			noiseCfg Attributes;

		protected:

			// Textures used for storing gradients and permutation table. (texture objects are always read-only)
			cudaTextureObject_t permutation;
		};

		class FBM3D {
		public:

		};

		class FBM4D {
		public:

		};
	}
}

#endif // !FBM_H
