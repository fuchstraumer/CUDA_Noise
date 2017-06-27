#include "ScaleBias.h"
#include "../cuda/modifiers/scalebias.cuh"

cnoise::modifiers::ScaleBias::ScaleBias(const size_t width, const size_t height, const float _scale, const float _bias) : Module(width, height), scale(_scale), bias(_bias){}

void cnoise::modifiers::ScaleBias::SetBias(const float _bias){
	bias = _bias;
}

void cnoise::modifiers::ScaleBias::SetScale(const float _scale){
	scale = _scale;
}

float cnoise::modifiers::ScaleBias::GetBias() const{
	return bias;
}

float cnoise::modifiers::ScaleBias::GetScale() const{
	return scale;
}

size_t cnoise::modifiers::ScaleBias::GetSourceModuleCount() const{
	return 1;
}

void cnoise::modifiers::ScaleBias::Generate(){
	if (sourceModules.front() == nullptr) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}
	scalebiasLauncher(Output, sourceModules.front()->Output, dims.first, dims.second, scale, bias);
	Generated = true;
}

cnoise::modifiers::ScaleBias3D::ScaleBias3D(Module3D* source, const int width, const int height, const float _scale, const float _bias) : Module3D(source, width, height), scale(_scale), bias(_bias) {}

void cnoise::modifiers::ScaleBias3D::SetBias(const float _bias) {
	bias = _bias;
}

void cnoise::modifiers::ScaleBias3D::SetScale(const float _scale) {
	scale = _scale;
}

float cnoise::modifiers::ScaleBias3D::GetBias() const {
	return bias;
}

float cnoise::modifiers::ScaleBias3D::GetScale() const {
	return scale;
}

size_t cnoise::modifiers::ScaleBias3D::GetSourceModuleCount() const {
	return 1;
}

void cnoise::modifiers::ScaleBias3D::Generate() {
	if (sourceModules.front() == nullptr) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}
	scalebiasLauncher3D(Points, sourceModules.front()->Points, dimensions.x, dimensions.y, scale, bias);
	Generated = true;
}
