#include "Terrace.h"
#include "../cuda/modifiers/terrace.cuh"

cnoise::modifiers::Terrace::Terrace(const int width, const int height) : Module(width, height), inverted(false) {}

void cnoise::modifiers::Terrace::Generate(){
	if (sourceModules.front() == nullptr || sourceModules.empty()) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}
	std::vector<float> points = GetControlPoints();
	// Launch kernel
	TerraceLauncher(Output, sourceModules.front()->Output, dims.first, dims.second, points, inverted);
	Generated = true;
}

size_t cnoise::modifiers::Terrace::GetSourceModuleCount() const{
	return 1;
}

void cnoise::modifiers::Terrace::AddControlPoint(const float & val){
	controlPoints.insert(val);
}

void cnoise::modifiers::Terrace::ClearControlPoints(){
	controlPoints.clear();
}

std::vector<float> cnoise::modifiers::Terrace::GetControlPoints() const{
	return std::vector<float>(controlPoints.begin(), controlPoints.end());
}

void cnoise::modifiers::Terrace::MakeControlPoints(const size_t & num_pts){
	ClearControlPoints();
	float step = 1.0f / num_pts;
	for (size_t i = 0; i < num_pts; ++i) {
		controlPoints.insert(step * i);
	}
}

void cnoise::modifiers::Terrace::SetInversion(bool inv){
	inverted = inv;
}

bool cnoise::modifiers::Terrace::GetInversion() const{
	return inverted;
}
