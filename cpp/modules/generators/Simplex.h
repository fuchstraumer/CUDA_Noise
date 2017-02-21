#pragma once
#ifndef SIMPLEX_H
#define SIMPLEX_H
#include "common\CommonInclude.h"
#include "common\CUDA_Include.h"
#include "..\Base.h"
/*

	Defines a simplex-noise generator base class
	for integration with the other fractal generators.

*/
namespace noise {
	namespace module {

		class Simplex2D : public Module {
		public:
			
			// Ctor
			Simplex2D(int width, int height, int seed);

		protected:

			// CUDA array pointers
			cudaArray *pArray, *gArray;

			// CUDA texture objects that will be the actual interface to the above
			// arrays (at least in the CUDA code itself).
			cudaTextureObject_t pTex, gTex;

		};
	}
}


#endif // !SIMPLEX_H
