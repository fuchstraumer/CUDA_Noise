#pragma once
#ifndef BILLOW_H
#define BILLOW_H
#include "Perlin.h"

namespace noise {
	namespace module {
		// Default parameters
		constexpr float DEFAULT_BILLOW_FREQUENCY = 1.0f;
		constexpr float DEFAULT_BILLOW_LACUNARITY = 2.0f;
		constexpr int DEFAULT_BILLOW_OCTAVES = 6;
		constexpr float DEFAULT_BILLOW_PERSISTENCE = 0.50f;
		constexpr int DEFAULT_BILLOW_SEED = 0;

		// Maximum octave level to allow
		constexpr int BILLOW_MAX_OCTAVES = 24;


		class Billow2D : public Perlin2D {
		public:

			// Width + height specify output texture size.
			// Seed defines a value to seed the generator with
			// X & Y define the origin of the noise generator
			Billow2D(int width, int height, int x = 0, int y = 0, int seed = DEFAULT_BILLOW_SEED, float freq = DEFAULT_BILLOW_FREQUENCY, float lacun = DEFAULT_BILLOW_LACUNARITY,
				int octaves = DEFAULT_BILLOW_OCTAVES, float persist = DEFAULT_BILLOW_PERSISTENCE);

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
	}
}


#endif // !BILLOW_H
