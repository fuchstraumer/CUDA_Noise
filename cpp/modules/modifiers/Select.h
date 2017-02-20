#pragma once
#ifndef SELECT_H
#define SELECT_H
#include "stdafx.h"
#include "Base.h"

namespace noise {
	namespace module {

		/*

			Select module:

			A selector/select module chooses which of two 
			connected subject modules to sample a value from 
			using a selector module.

			For this other module, a range of values is set 
			(along with an optional falloff value), and if 
			the value falls within this range subject0 is chosen.
			If it is outside of the range, subject1 is chosen.

			The falloff value eases the transition between
			the two module's values in the final output.
		
		*/


		class Select : public Module {
		public:

			// The falloff value does not have to be set at all. The Modules MUST be set eventually, but don't need to be set upon initialization.
			Select(int width, int height, float low_value, float high_value, float falloff = 0.15f, std::shared_ptr<Module> selector = nullptr, std::shared_ptr<Module> subject0 = nullptr, std::shared_ptr<Module> subject1 = nullptr);

		};

	}
}
#endif // !SELECT_H
