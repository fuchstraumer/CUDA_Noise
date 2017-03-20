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


namespace cnoise {

	constexpr float DEGREES_TO_RADIANS = 3.141592653589f / 180.0f;

	typedef struct alignas(sizeof(float)) Point {
		float3 Position;
		float Value;
		Point(float x, float y, float z) : Position(make_float3(x, y, z)), Value(0.0f) {}
		Point() : Value(0.0f) {}
	} Point;

	typedef struct GeoCoord : public Point {
		// Create geocoord from lattitude/longitude points.
		GeoCoord(float lattitude, float longitude) {
			float r = std::cosf(DEGREES_TO_RADIANS * lattitude);
			Position.x = r * std::cosf(DEGREES_TO_RADIANS * longitude);
			Position.y = std::sinf(DEGREES_TO_RADIANS * lattitude);
			Position.z = r * std::sinf(DEGREES_TO_RADIANS * longitude);
		}
	} GeoCoord;

}
#endif // !CUDA_INCLUDE_H
