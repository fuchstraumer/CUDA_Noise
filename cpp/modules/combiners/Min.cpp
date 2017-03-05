#include "Min.h"
#include "../cuda/combiners/min.cuh"

cnoise::combiners::Min::Min(const int width, const int height, Module * in0, Module * in1) : Module(width, height) {
	sourceModules.push_back(in0);
	sourceModules.push_back(in1);
}

void cnoise::combiners::Min::Generate(){
	for (const auto m : sourceModules) {
		if (m == nullptr) {
			throw;
		}
		if (!m->Generated) {
			m->Generate();
		}
	}
	MinLauncher(Output, sourceModules[0]->Output, sourceModules[1]->Output, dims.first, dims.second);
	Generated = true;
}

size_t cnoise::combiners::Min::GetSourceModuleCount() const{
	return 2;
}
