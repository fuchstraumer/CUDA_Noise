#ifndef TERRACE_CUH
#define TERRACE_CUH
#include "../common/CUDA_Include.h"

void TerraceLauncher(float* output, const float* input, const int width, const int height, const std::vector<float>& pts, bool invert);

void TerraceLauncher3D(cnoise::Point* data, const int width, const int height, const std::vector<float>& pts, bool invert);

#endif // !TERRACE_CUH
