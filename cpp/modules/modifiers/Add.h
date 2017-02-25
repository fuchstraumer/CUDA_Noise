#pragma once
#ifndef ADD_H
#define ADD_H

#include "common\CommonInclude.h"
#include "common\CUDA_Include.h"

#include "../Base.h"

namespace noise {

	namespace module {

		class Add : public Module {
		public:

			Add(int width, int height, float add_value);
		};

	}

}

#endif // !ADD_H
