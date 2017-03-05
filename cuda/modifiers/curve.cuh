#ifndef CURVE_CUH
#define CURVE_CUH
#include "../common/CUDA_Include.h"

void CurveLauncher(float* output, float* input, const int width, const int height, std::vector<ControlPoint>& control_points);

#endif // !CURVE_CUH
