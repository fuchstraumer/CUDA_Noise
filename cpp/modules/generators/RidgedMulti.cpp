#include "RidgedMulti.h"
#include "../cuda/generators/ridged_multi.cuh"
namespace cnoise {

	namespace generators {


		RidgedMulti::RidgedMulti(int width, int height, noise_t noise_type, float x, float y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height),
			Attributes(seed, freq, lacun, octaves, persist), Origin(x, y), NoiseType(noise_type) {}

		size_t RidgedMulti::GetSourceModuleCount() const {
			return 0;
		}

		void RidgedMulti::Generate(){
			RidgedMultiLauncher(Output, dims.first, dims.second, NoiseType, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			Generated = true;
		}

		RidgedMulti3D::RidgedMulti3D(int width, int height, int seed, float freq, float lacun, int octaves, float persist) : Attributes(seed, freq, lacun, octaves, persist), Module3D(nullptr, width, height) {}

		size_t RidgedMulti3D::GetSourceModuleCount() const{
			return 0;
		}

		void RidgedMulti3D::Generate(){
			RidgedMultiLauncher3D(Points, dimensions.x, dimensions.y, Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			Generated = true;
		}

	}

}