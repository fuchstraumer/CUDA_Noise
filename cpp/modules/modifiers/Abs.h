#pragma once
#ifndef ABS_H
#define ABS_H
#include "../Base.h"

namespace cnoise {

	namespace modifiers {

		class Abs : public Module {
		public:

			Abs(const size_t width, const size_t height, Module* previous);

			virtual size_t GetSourceModuleCount() const override;

			void Generate() override;

		};

	}

}

#endif // !ABS_H
