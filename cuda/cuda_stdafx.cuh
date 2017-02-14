#ifndef CUDA_STDAFX_H
#define CUDA_STDAFX_H

/*
	CUDA_STDAFX_H

	Standard includes and #define's for use with CUDA
	code specifically - this was seperated as CUDA doesn't
	seem to support some kind of precompiled header, whereas
	C++ does 

*/

// Need this for GL types from GLEW
#include "stdafx.h"

// CUDA includes.
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda_gl_interop.h>
#include <cuda_surface_types.h>
#include <vector_types.h>
#include <device_functions.h>

// alias declaration so I can be lazier about using unsigned ints
using uint = unsigned int;

#endif // !CUDA_STDAFX_H
