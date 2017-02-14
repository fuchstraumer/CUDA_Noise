#pragma once
#ifndef PERLIN_H
#define PERLIN_H
#include "cuda_stdafx.cuh"
#include "Base.h"
/*

	Generates values using perlin noise. Ctor includes desired dimensions
	of output texture.
	
	These classes must exist as wrappers over the CUDA kernels since we need
	to allocate and create our constant texture objects, used as permutation
	tables and the like. Without these, we have to use costly/slow array lookups.

*/
namespace noise {

	namespace module {

		// Default parameters
		constexpr float DEFAULT_PERLIN_FREQUENCY = 1.0f;
		constexpr float DEFAULT_PERLIN_LACUNARITY = 2.0f;
		constexpr int DEFAULT_PERLIN_OCTAVES = 6;
		constexpr float DEFAULT_PERLIN_PERSISTENCE = 0.50f;
		constexpr int DEFAULT_PERLIN_SEED = 0;

		// Maximum octave level to allow
		constexpr int PERLIN_MAX_OCTAVES = 24;


		class Perlin2D : public Module {
		public:

			// Width + height specify output texture size.
			// Seed defines a value to seed the generator with
			// X & Y define the origin of the noise generator
			Perlin2D(int width, int height, int x = 0, int y = 0, int seed = DEFAULT_PERLIN_SEED, float freq = DEFAULT_PERLIN_FREQUENCY, float lacun = DEFAULT_PERLIN_LACUNARITY,
				int octaves = DEFAULT_PERLIN_OCTAVES, float persist = DEFAULT_PERLIN_PERSISTENCE);

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

		class Perlin3D {
		public:

		};

		class Perlin4D {
		public:

		};
	}
}

#endif // !PERLIN_H
