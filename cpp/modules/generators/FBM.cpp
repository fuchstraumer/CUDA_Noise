#include "FBM.h"
#include "..\cuda\generators\FBM.cuh"
#include "..\cuda\cuda_assert.h"
namespace noise {
	
	namespace module {

		// Pass width and height to base class ctor, initialize configuration struct, initialize origin (using initializer list)
		FBM2D::FBM2D(int width, int height, int x, int y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height), 
			Attributes(seed, freq, lacun, octaves, persist), Origin(x,y) {}

		// TODO: Implement these. Just here so compiler shuts up.
		int FBM2D::GetSourceModuleCount() const{
			return 0;
		}

		void FBM2D::Generate(){
			FBM_Launcher(output, dims.first, dims.second, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			Generated = true;
		}

		// Pass width and height to base class ctor, initialize configuration struct, initialize origin (using initializer list)
		FBM2DSimplex::FBM2DSimplex(int width, int height, int x, int y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height),
			Attributes(seed, freq, lacun, octaves, persist), Origin(x, y) {}

		// TODO: Implement these. Just here so compiler shuts up.
		int FBM2DSimplex::GetSourceModuleCount() const {
			return 0;
		}

		void FBM2DSimplex::Generate() {
			FBM_Launcher_Simplex(output, dims.first, dims.second, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			Generated = true;
		}
	}
}
