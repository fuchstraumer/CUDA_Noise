#include "Constant.h"

cnoise::utility::Constant::Constant(const int width, const int height, const float value) : Module(width, height) {
	cudaError_t err = cudaSuccess;
	err = cudaDeviceSynchronize();
	cudaAssert(err);
	std::vector<float> constant_val;
	constant_val.assign(width * height, value);
	err = cudaMemcpy(Output, &constant_val[0], width * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaAssert(err);
	err = cudaDeviceSynchronize();
	cudaAssert(err);
	Generated = true;
}

size_t cnoise::utility::Constant::GetSourceModuleCount() const{
	return 0;
}

void cnoise::utility::Constant::Generate(){}
