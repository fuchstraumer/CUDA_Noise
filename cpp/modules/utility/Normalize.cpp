#include "Normalize.h"
#include "../cuda/utility/normalize.cuh"
cnoise::utility::Normalize::Normalize(int width, int height, Module * source) : Module(width, height){
	sourceModules.push_back(source);
}

void cnoise::utility::Normalize::Generate(){
	if (sourceModules.front() == nullptr || sourceModules.empty()) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}
	NormalizeLauncher(Output, sourceModules.front()->Output, dims.first, dims.second);
	Generated = true;
}

size_t cnoise::utility::Normalize::GetSourceModuleCount() const{
	return 1;
}
