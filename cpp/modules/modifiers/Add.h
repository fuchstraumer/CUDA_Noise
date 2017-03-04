#pragma once
#ifndef ADD_H
#define ADD_H
#include "../Base.h"

namespace noise {

	namespace module {

		class Add : public Module {
		public:

			Add(int width, int height, float add_value, Module* source = nullptr);

			virtual void Generate() override;

			virtual size_t GetSourceModuleCount() const override;

			// Set value to add to source module
			void SetAddValue(float val);

			// Get add value
			float GetAddValue() const;

		private: 

			float addValue;
		};

	}

}

#endif // !ADD_H
