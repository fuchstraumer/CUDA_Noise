#include "Billow.h"

// Include cuda modules
#include "..\cuda\generators\billow.cuh"

namespace noise {
	namespace module {

		Billow2D::Billow2D(int width, int height, float x, float y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height), Attributes(seed, freq, lacun, octaves, persist), Origin(x, y) {}

		int Billow2D::GetSourceModuleCount() const {
			return 0;
		}

		void Billow2D::Generate(){
			BillowLauncher(output, dims.first, dims.second, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			Generated = true;
		}

		Billow2DSimplex::Billow2DSimplex(int width, int height, float x, float y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height),
			Attributes(seed, freq, lacun, octaves, persist), Origin(x, y) {}

		int Billow2DSimplex::GetSourceModuleCount() const {
			return 0;
		}

		void Billow2DSimplex::Generate() {
			BillowSimplexLauncher(output, dims.first, dims.second, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			Generated = true;
		}
	}
}
