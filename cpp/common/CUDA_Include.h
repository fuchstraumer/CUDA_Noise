#pragma once
#ifndef CUDA_INCLUDE_H
#define CUDA_INCLUDE_H
#include "CommonInclude.h"
/*

	CUDA_INCLUDE_H

	Used for including the required CUDA components in C++.
	
*/
#define CUDA_TIMING_TESTS
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <device_functions.h>
#include "../cuda/cuda_assert.h"


enum noise_t {
	PERLIN,
	SIMPLEX,
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
