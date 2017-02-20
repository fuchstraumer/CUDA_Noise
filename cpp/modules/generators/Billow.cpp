#include "Billow.h"

// Include cuda modules
#include "..\cuda\generators\billow.cuh"

namespace noise {
	namespace module {

		Billow2D::Billow2D(int width, int height, float x, float y, int seed, float freq, float lacun, int octaves, float persist) : Perlin2D(width, height),
			Attributes(freq, lacun, persist, octaves, seed, BILLOW_MAX_OCTAVES), Origin(x, y) {}

		int Billow2D::GetSourceModuleCount() const {
			return 0;
		}

		void Billow2D::Generate(){
			BillowLauncher(output, permTex, gradTex, dims.first, dims.second, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
		}

	}
}
