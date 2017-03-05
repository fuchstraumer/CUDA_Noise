#include "Power.h"
#include "../cuda/combiners/power.cuh"

cnoise::combiners::Power::Power(const int width, const int height, Module * in0, Module * in1) : Module(width, height){
	sourceModules.push_back(in0);
	sourceModules.push_back(in1);
}

void cnoise::combiners::Power::Generate(){
	for (const auto& m : sourceModules) {
		if (m == nullptr) {
			throw;
		}
		if (!m->Generated) {
			m->Generate();
		}
	}
	powerLauncher(Output, sourceModules[0]->Output, sourceModules[1]->Output, dims.first, dims.second);
	Generated = true;
}

size_t cnoise::combiners::Power::GetSourceModuleCount() const{
	return 2;
}
