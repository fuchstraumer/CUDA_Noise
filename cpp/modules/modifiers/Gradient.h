#pragma once
#ifndef GRADIENT_H
#define GRADIENT_H
#include "common/CommonInclude.h"
#include "../Base.h"

namespace noise {

	namespace module {

		class Gradient2D : public Module {
		public:

			// Constructor defaults to assuming simple diagonal gradient.
			Gradient2D(int width, int height, float x0 = 0.0f, float x1 = 1.0f, float y0 = 0.0f, float y1 = 1.0f);

			// Set gradient using given values.
			float SetGradient(float x0, float x1, float y0, float y1);

			std::array<float, 4> GetGradient() const;

		protected:

			// Holds values defining the actual gradient.
			std::array<float, 4> gradient;
		};

	}
}



#endif // !GRADIENT_H
