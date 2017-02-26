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

typedef struct alignas(sizeof(float)) ControlPoint {
	float InputVal, OutputVal;
	ControlPoint(float in, float out) : InputVal(in), OutputVal(out) {}
} ControlPoint;

#endif // !COMMON_INCLUDE_H
