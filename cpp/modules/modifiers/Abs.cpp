#include "Abs.h"
#include "../cuda/modifiers/abs.cuh"

cnoise::modifiers::Abs::Abs(const size_t width, const size_t height, Module * previous) : Module(width, height) {
	sourceModules.push_back(std::shared_ptr<Module>(previous));
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
	absLauncher(Output, sourceModules.front()->Output, dims.first, dims.second);
	Generated = true;
}

size_t cnoise::modifiers::Abs3D::GetSourceModuleCount() const{
	return 1;
}

void cnoise::modifiers::Abs3D::Generate(){
	if (sourceModules.front() == nullptr || sourceModules.empty()) {
		throw;
	}
	else {
		absLauncher3D(Points, sourceModules.front()->Points, dimensions.x, dimensions.y);
		Generated = true;
	}
}
