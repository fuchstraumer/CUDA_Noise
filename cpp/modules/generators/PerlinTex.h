#pragma once
#ifndef PERLIN_TEX_H
#define PERLIN_TEX_H
#include "../Base.h"

namespace noise {
	namespace module {

		class PerlinTexBase : public Module {
		public:

			PerlinTexBase(int width, int height, float2 origin, int seed);

			virtual void Generate() override;

			virtual int GetSourceModuleCount() const override;

		private:

			cudaArray *gradientArray, *permutationArray;

			cudaTextureObject_t gradient, permutation;

		};
	}
}



#endif // !PERLIN_TEX_H
