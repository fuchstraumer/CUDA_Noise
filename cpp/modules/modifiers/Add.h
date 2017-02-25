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

			Add(int width, int height, float add_value, std::shared_ptr<Module> source = nullptr);

			virtual void Generate() override;

			virtual int GetSourceModuleCount() const override;

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
