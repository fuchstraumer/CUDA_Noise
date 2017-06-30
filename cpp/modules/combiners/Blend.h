#pragma once
#ifndef BLEND_H
#define BLEND_H
#include "../Base.h"

namespace cnoise {

	namespace combiners {

		class Blend : public Module {
		public:

			Blend(const int width, const int height, Module* in0 = nullptr, Module* in1 = nullptr, Module* weight_module = nullptr);

			virtual void Generate() override;

			virtual size_t GetSourceModuleCount() const override;

			void SetSourceModule(const int idx, Module* source);

			void SetControlModule(Module* control);

		};

		class Blend3D : public Module3D {

			Blend3D(Module3D* left, Module3D* right, Module3D* weight_module, const int& width, const int& height);

			virtual size_t GetSourceModuleCount() const override;
			virtual void Generate() override;
		};

	}

}
#endif // !BLEND_H
