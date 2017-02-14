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
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <device_functions.h>


// alias declaration so I can be lazier about using unsigned ints
using uint = unsigned int;

// Check CUDA version and decide if we'll enable half-precision
// TODO: Disabled this until base version working. make sure to come back to it later.
/*#if CUDART_VERSION >= 7050
#define HALF_PRECISION_SUPPORT
#endif

#ifdef HALF_PRECISION_SUPPORT
#include <cuda_fp16.h>
#endif // HALF_PRECISION_SUPPORT*/


#endif // !CUDA_STDAFX_H
