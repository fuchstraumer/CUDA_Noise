#pragma once
#ifndef CUDA_INCLUDE_H
#define CUDA_INCLUDE_H
#include "CommonInclude.h"
/*

	CUDA_INCLUDE_H

	Used for including the required CUDA components in C++.
	
*/
#define CUDA_KERNEL_TIMING
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <device_functions.h>
#include "../cuda/cuda_assert.h"

typedef unsigned int uint;
typedef unsigned char uchar;

enum noise_t {
	PERLIN,
	SIMPLEX,
};

// Type of distance function to use in voronoi generation
enum voronoi_distance_t {
	MANHATTAN,
	EUCLIDEAN,
	CELLULAR,
};

// Type of value to get from a voronoi function, and then store in the output texture.
enum voronoi_return_t {
	CELL_VALUE, // Get cell coord/val. Analagous to value noise.
	NOISE_LOOKUP, // Use coords to get a noise value
	DISTANCE, // Get distance to node.
};

enum noise_quality {
	FAST,
	STANDARD,
	HIGH,
};

typedef struct alignas(sizeof(float)) ControlPoint {
	float InputVal, OutputVal;
	ControlPoint(float in, float out) : InputVal(in), OutputVal(out) {}
} ControlPoint;

#endif // !CUDA_INCLUDE_H
