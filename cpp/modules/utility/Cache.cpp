#include "Cache.h"

cnoise::utility::Cache::Cache(int width, int height, Module * source) : Module(width, height) {
	sourceModules.push_back(std::shared_ptr<Module>(source));
}

void cnoise::utility::Cache::Generate(){
	if (sourceModules.front() == nullptr) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}

	cudaAssert(cudaDeviceSynchronize());

	cudaAssert(cudaMemcpy(Output, sourceModules.front()->Output, sizeof(sourceModules.front()->Output), cudaMemcpyDefault));

	cudaAssert(cudaDeviceSynchronize());
	//sourceModules.front()->~Module();
}

size_t cnoise::utility::Cache::GetSourceModuleCount() const{
	return 1;
}

cnoise::utility::Cache3D::Cache3D(int width, int height, Module3D * source) : Module3D(source, width, height) {
	sourceModules.push_back(std::shared_ptr<Module3D>(source));
}

void cnoise::utility::Cache3D::Generate() {
	if (sourceModules.front() == nullptr) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}
	// We make sure that all previous modules have been generated by synchronizing (esp. important in the case of
	// 3D modules, which use managed memory), then copy the front modules data to this object and then tell CUDA
	// to free/delete the memory belonging to the rest of the modules.
	cudaError_t err = cudaSuccess;
	err = cudaDeviceSynchronize();
	cudaAssert(err);
	err = cudaMemcpy(Points, sourceModules.front()->Points, dimensions.x * dimensions.y * sizeof(Point), cudaMemcpyDefault);
	cudaAssert(err);
	// Now deallocate all connected modules recursively.
	this->freeChildren();
}

size_t cnoise::utility::Cache3D::GetSourceModuleCount() const {
	return 1;
}
