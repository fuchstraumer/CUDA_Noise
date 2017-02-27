#pragma once
#ifndef RIDGED_MULTI_H
#define RIDGED_MULTI_H
#include "common\CUDA_Include.h"
#include "../Base.h"

namespace noise {

	namespace module {

		// Default parameters
		constexpr float DEFAULT_RIDGED_FREQUENCY = 1.0f;
		constexpr float DEFAULT_RIDGED_LACUNARITY = 2.0f;
		constexpr int DEFAULT_RIDGED_OCTAVES = 6;
		constexpr float DEFAULT_RIDGED_PERSISTENCE = 0.50f;
		constexpr int DEFAULT_RIDGED_SEED = 0;

		// Maximum octave level to allow
		constexpr int RIDGED_MAX_OCTAVES = 24;

		class RidgedMulti : public Module {
		public:
			// Width + height specify output texture size.
			// Seed defines a value to seed the generator with
			// X & Y define the origin of the noise generator
			RidgedMulti(int width, int height, noise_t noise_type = noise_t::PERLIN, int x = 0, int y = 0, int seed = DEFAULT_RIDGED_SEED, float freq = DEFAULT_RIDGED_FREQUENCY, float lacun = DEFAULT_RIDGED_LACUNARITY,
				int octaves = DEFAULT_RIDGED_OCTAVES, float persist = DEFAULT_RIDGED_PERSISTENCE);

			// Get source module count: must be 0, this is a generator and can't have preceding modules.
			virtual int GetSourceModuleCount() const override;

			// Launches the kernel and fills this object's surface object with the relevant data.
			virtual void Generate() override;

			// Origin of this noise generator. Keep the seed constant and change this for 
			// continuous "tileable" noise
			std::pair<float, float> Origin;

			// Configuration attributes.
			noiseCfg Attributes;

			// Type of noise to use.
			noise_t NoiseType;
		};

	}

}
#endif // !RIDGED_MULTI_H