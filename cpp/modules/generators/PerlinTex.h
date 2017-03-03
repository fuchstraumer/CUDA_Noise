#pragma once
#ifndef PERLIN_TEX_H
#define PERLIN_TEX_H
#include "../Base.h"

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

		class PerlinTexBase : public Module {
		public:

			PerlinTexBase(int width, int height, float origin_x, float origin_y, int seed = DEFAULT_PERLIN_SEED, float freq = DEFAULT_PERLIN_FREQUENCY, float lacun = DEFAULT_PERLIN_LACUNARITY,
				int octaves = DEFAULT_PERLIN_OCTAVES, float persist = DEFAULT_PERLIN_PERSISTENCE);

			virtual void Generate() override;

			virtual int GetSourceModuleCount() const override;

			noiseCfg Attributes;

			std::pair<float, float> Origin;

		private:



			cudaArray *gradientArray, *permutationArray;

			cudaTextureObject_t gradient, permutation;

		};
	}
}



#endif // !PERLIN_TEX_H
