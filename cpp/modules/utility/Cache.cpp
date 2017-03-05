#include "Cache.h"

cnoise::utility::Cache::Cache(int width, int height, Module * source) : Module(width, height) {
	sourceModules.push_back(source);
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
