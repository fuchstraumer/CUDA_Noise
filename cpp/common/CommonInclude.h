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

#endif // !COMMON_INCLUDE_H
