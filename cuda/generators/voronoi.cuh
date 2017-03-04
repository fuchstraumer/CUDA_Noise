#ifndef VORONOI_CUH
#define VORONOI_CUH
#include "../common/CUDA_Include.h"

void VoronoiLauncher(cudaSurfaceObject_t out, const int width, const int height, const float freq, const float displacement, const voronoi_distance_t dist_func, const voronoi_return_t return_t);

#endif // !VORONOI_CUH
