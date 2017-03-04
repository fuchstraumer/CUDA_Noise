#include "Abs.h"
#include "../cuda/modifiers/abs.cuh"

cnoise::modifiers::Abs::Abs(const size_t width, const size_t height, Module * previous) : Module(width, height) {
}

size_t cnoise::modifiers::Abs::GetSourceModuleCount() const{
	return 1;
}

void cnoise::modifiers::Abs::Generate() {
	if (sourceModules.front() == nullptr) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}
	absLauncher(output, sourceModules.front()->output, dims.first, dims.second);
	Generated = true;
}