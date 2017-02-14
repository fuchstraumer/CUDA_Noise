#pragma once

 // Common includes for C++ elements of this program.

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

// OpenGL includes

// GLEW (must come first)
// Define as static so no need to package DLL in this repo.
#define GLEW_STATIC
#include "GL\glew.h"

// GLFW
#include "GLFW\glfw3.h"

// GLM
#define GLM_SWIZZLE
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "glm\gtc\type_ptr.hpp"

// Set screen width and height
static constexpr int SCR_WIDTH = 1440, SCR_HEIGHT = 720;

// Threads to use per cuda block. can change for tuning reasons.
constexpr int threads_per_block = 128;