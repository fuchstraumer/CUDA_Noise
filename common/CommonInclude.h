#pragma once
#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H
/*
	
	COMMON_INCLUDE_H

	Defines common include's required by most of the C++ files
	in this program.

*/

// Standard library includes.

#include <vector>
#include <iostream>
#include <array>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <memory>

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

typedef unsigned int uint;
typedef unsigned char uchar;

typedef struct alignas(sizeof(float)) ControlPoint {
	float InputVal, OutputVal;
	ControlPoint(float in, float out) : InputVal(in), OutputVal(out) {}
} ControlPoint;



#endif // !COMMON_INCLUDE_H
