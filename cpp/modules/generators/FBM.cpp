#include "FBM.h"
#include "..\cuda\generators\perlin.cuh"
#include "..\cuda\cuda_assert.h"
namespace noise {
	
	namespace module {

		// Pass width and height to base class ctor, initialize configuration struct, initialize origin (using initializer list)
		FBM2D::FBM2D(int width, int height, int x, int y, int seed, float freq, float lacun, int octaves, float persist) : Perlin2D(width, height), 
			Attributes(freq, lacun, persist, octaves, seed, FBM_MAX_OCTAVES), Origin(x,y) {}

		// TODO: Implement these. Just here so compiler shuts up.
		int FBM2D::GetSourceModuleCount() const{
			return 0;
		}

		void FBM2D::Generate(){
			// Need to remake this.
		}
	}
}
