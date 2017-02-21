#pragma once
#ifndef PERLIN_H
#define PERLIN_H
#include "common\CommonInclude.h"
#include "..\Base.h"

namespace noise {
	namespace module {
		/*

		Base Module: Perlin noise

		This is a base module, as it acts as the base noise generator
		from which other modules will draw.

		This is because we must generate two texture objects that we
		will pass into the kernels. Like how the CUDA files rely on
		the base Perlin.cu/cuh file, all the other noise modules
		in this library rely on this module. It takes care of the common
		tasks, like setting up these lookup objects and instantiating the
		base class, but does not launch the kernel. This is left
		to those derived classes. This class should throw errors about
		instantiating an abstract class if one tries to use it directly.

		Note: The base class is currently called Perlin2D, as I'd like to
		integrate higher-dimensionality noise down the road.

		*/

		class Perlin2D : public Module {
		public:

			Perlin2D(int width, int height, int seed);

			~Perlin2D();

			// Methods are all virtual, still, since we haven't overriden them.

		protected:

			// initial perm char table used to generate textures.
			unsigned char perm[256];

			// Array for permutation and gradient lookups (textures)
			cudaArray *permArray, *gradArray;

			// Textures for permutation and gradient lookups
			cudaTextureObject_t permTex, gradTex;

		};
	}

}

#endif // !PERLIN_H
