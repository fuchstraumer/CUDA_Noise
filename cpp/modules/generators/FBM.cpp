#include "FBM.h"
#include "..\cuda\generators\FBM.cuh"
#include "..\cuda\cuda_assert.h"
namespace noise {
	
	namespace module {

		// Pass width and height to base class ctor, initialize configuration struct, initialize origin (using initializer list)
		FBM2D::FBM2D(int width, int height, noise_t noise_type, int x, int y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height), 
			Attributes(seed, freq, lacun, octaves, persist), Origin(x,y), NoiseType(noise_type) {}

		// TODO: Implement these. Just here so compiler shuts up.
		size_t FBM2D::GetSourceModuleCount() const{
			return 0;
		}

		void FBM2D::Generate(){
			FBM_Launcher(output, dims.first, dims.second, NoiseType, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			Generated = true;
		}
	}
}
