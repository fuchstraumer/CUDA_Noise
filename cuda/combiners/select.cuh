#ifndef SELECT_CUH
#define SELECT_CUH
#include "../common/CUDA_Include.h"

void SelectLauncher(float* out, float* select_item, float* subject0, float* subject1, int width, int height, float upper_bound, float lower_bound, float falloff);

void SelectLauncher3D(cnoise::Point* left, const cnoise::Point* right, const cnoise::Point* select_data, const int width, const int height, const float upper_bound, const float lower_bound, const float& falloff);

#endif // !SELECT_CUH