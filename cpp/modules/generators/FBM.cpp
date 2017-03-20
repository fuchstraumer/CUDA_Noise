#include "FBM.h"
#include "..\cuda\generators\FBM.cuh"
#include "..\cuda\cuda_assert.h"
namespace cnoise {
	
	namespace generators {

		// Pass width and height to base class ctor, initialize configuration struct, initialize origin (using initializer list)
		FBM2D::FBM2D(int width, int height, noise_t noise_type, float x, float y, int seed, float freq, float lacun, int octaves, float persist) : Module(width, height), 
			Attributes(seed, freq, lacun, octaves, persist), Origin(x,y), NoiseType(noise_type) {}

		// TODO: Implement these. Just here so compiler shuts up.
		size_t FBM2D::GetSourceModuleCount() const{
			return 0;
		}

		void FBM2D::Generate(){
			FBM_Launcher(Output, dims.first, dims.second, NoiseType, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
			Generated = true;
		}
	}
}

cnoise::generators::FBM3D::FBM3D(int width, int height, int seed, float freq, float lacun, int octaves, float persist) : Attributes(seed, freq, lacun, octaves, persist), Module3D(width, height) {}

size_t cnoise::generators::FBM3D::GetSourceModuleCount() const{
	return 0;
}

void cnoise::generators::FBM3D::Generate(){
	FBM_Launcher3D(Points, dimensions.x, dimensions.y, Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
	Generated = true;
}

