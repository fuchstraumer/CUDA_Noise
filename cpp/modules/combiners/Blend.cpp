#include "Blend.h"
#include "../cuda/combiners/blend.cuh"

cnoise::combiners::Blend::Blend(const int width, const int height, Module * in0, Module * in1, Module * weight_module) : Module(width, height) {
	sourceModules.push_back(in0);
	sourceModules.push_back(in1);
	sourceModules.push_back(weight_module);
}

void cnoise::combiners::Blend::Generate() {
	for (const auto m : sourceModules) {
		if (m == nullptr) {
			throw;
		}
		if (!m->Generated) {
			m->Generate();
		}
	}
	BlendLauncher(Output, sourceModules[0]->Output, sourceModules[1]->Output, sourceModules[2]->Output, dims.first, dims.second);
	Generated = true;
}

size_t cnoise::combiners::Blend::GetSourceModuleCount() const{
	return 3;
}

void cnoise::combiners::Blend::SetSourceModule(const int idx, Module * source){
	if (idx > 2 || idx < 0) {
		return;
	}
	sourceModules[idx] = source;
}

void cnoise::combiners::Blend::SetControlModule(Module * control){
	sourceModules[2] = control;
}
