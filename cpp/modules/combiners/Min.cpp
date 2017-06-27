#include "Min.h"
#include "../cuda/combiners/min.cuh"

cnoise::combiners::Min::Min(const int width, const int height, Module * in0, Module * in1) : Module(width, height) {
	sourceModules.push_back(std::shared_ptr<Module>(in0));
	sourceModules.push_back(std::shared_ptr<Module>(in1));
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

cnoise::combiners::Min3D::Min3D(const int width, const int height, Module3D * in0, Module3D * in1) : Module3D(in0, in1, width, height) {
	sourceModules.push_back(std::shared_ptr<Module3D>(in0));
	sourceModules.push_back(std::shared_ptr<Module3D>(in1));
}

void cnoise::combiners::Min3D::Generate() {
	for (const auto m : sourceModules) {
		if (m == nullptr) {
			throw;
		}
		if (!m->Generated) {
			m->Generate();
		}
	}
	MinLauncher3D(Points, sourceModules[0]->Points, sourceModules[1]->Points, dimensions.x, dimensions.y);
	Generated = true;
}

size_t cnoise::combiners::Min3D::GetSourceModuleCount() const {
	return 2;
}
