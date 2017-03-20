#include "Clamp.h"
#include "../cuda/modifiers/clamp.cuh"

cnoise::modifiers::Clamp::Clamp(int width, int height, float lower_bound, float upper_bound, Module * source) : Module(width, height), lowerBound(lower_bound), upperBound(upper_bound) {
	ConnectModule(source);
}

size_t cnoise::modifiers::Clamp::GetSourceModuleCount() const{
	return 1;
}

void cnoise::modifiers::Clamp::Generate(){
	if (sourceModules.front() == nullptr) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}
	ClampLauncher(Output, sourceModules.front()->Output, dims.first, dims.second, lowerBound, upperBound);
	Generated = true;
}

float cnoise::modifiers::Clamp::GetLowerBound() const{
	return lowerBound;
}

float cnoise::modifiers::Clamp::GetUpperBound() const{
	return upperBound;
}

void cnoise::modifiers::Clamp::SetLowerBound(const float lower){
	lowerBound = lower;
}

void cnoise::modifiers::Clamp::SetUpperBound(const float upper){
	upperBound = upper;
}

cnoise::modifiers::Clamp3D::Clamp3D(int width, int height, float lower_bound, float upper_bound, Module3D * source) : Module3D(width, height) {
	sourceModules.push_back(source);
}

size_t cnoise::modifiers::Clamp3D::GetSourceModuleCount() const{
	return 1;
}

void cnoise::modifiers::Clamp3D::Generate() {
	if (sourceModules.front() == nullptr) {
		throw;
	}
	else {
		if (!sourceModules.front()->Generated) {
			sourceModules.front()->Generate();
		}
		ClampLauncher3D(Points, sourceModules.front()->Points, dimensions.x, dimensions.y, lowerBound, upperBound);
	}
	Generated = true;
}

float cnoise::modifiers::Clamp3D::GetLowerBound() const{
	return lowerBound;
}

float cnoise::modifiers::Clamp3D::GetUpperBound() const{
	return upperBound;
}

void cnoise::modifiers::Clamp3D::SetLowerBound(const float lower){
	lowerBound = lower;
}

void cnoise::modifiers::Clamp3D::SetUpperBound(const float upper){
	upperBound = upper;
}
