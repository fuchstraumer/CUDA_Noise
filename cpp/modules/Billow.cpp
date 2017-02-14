#include "stdafx.h"
#include "Billow.h"

// Include cuda modules
#include "..\cuda\generators\billow.cuh"

namespace noise {
	namespace module {

		Billow2D::Billow2D(int width, int height, int x, int y, int seed, float freq, float lacun, int octaves, float persist) : Perlin2D(width, height),
			Attributes(freq, lacun, persist, octaves, seed, BILLOW_MAX_OCTAVES), Origin(x, y) {}

		int Billow2D::GetSourceModuleCount() const {
			return 0;
		}

		void Billow2D::Generate(){}

	}
}
