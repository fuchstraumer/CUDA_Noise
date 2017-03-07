#include "Checkerboard.h"
#include "../cuda/utility/checkerboard.cuh"

cnoise::utility::Checkerboard::Checkerboard(const int width, const int height) : Module(width, height) {}

void cnoise::utility::Checkerboard::Generate(){
	CheckerboardLauncher(Output, dims.first, dims.second);
	Generated = true;
}

size_t cnoise::utility::Checkerboard::GetSourceModuleCount() const{
	return 0;
}
