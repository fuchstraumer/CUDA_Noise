#include "Billow.h"

// Include cuda modules
#include "..\cuda\generators\billow.cuh"

namespace cnoise {

		namespace generators {

			Billow2D::Billow2D(int width, int height, noise_t noise_type, float x, float y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height), Attributes(seed, freq, lacun, octaves, persist), Origin(x, y), NoiseType(noise_type) {}

			size_t Billow2D::GetSourceModuleCount() const {
				return 0;
			}

			void Billow2D::Generate() {
				BillowLauncher2D(Output, dims.first, dims.second, NoiseType, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
				Generated = true;
			}

		}

}

cnoise::generators::Billow3D::Billow3D(int width, int height, int depth, float x, float y, float z, int seed, float freq, float lacun, int octaves, float persist) : Attributes(seed, freq, lacun, octaves, persist), Module3D(width, height) {}

size_t cnoise::generators::Billow3D::GetSourceModuleCount() const {
	return 0;
}

void cnoise::generators::Billow3D::Generate() {
	BillowLauncher3D(Points, dimensions.x, dimensions.y, Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
	Generated = true;
}