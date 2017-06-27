#pragma once
#ifndef CUDA_ASSERT_H
#define CUDA_ASSERT_H
#include "cuda_runtime.h"
#include <stdio.h>
#ifdef NDEBUG

// Runtime error-checker

#define cudaAssert(expression) { cudaErrCheck((expression), __FILE__, __LINE__); }

inline void cudaErrCheck(cudaError_t err, const char* file, unsigned line, bool abort = true) {
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaAssert: error %s at %s, Line %d \n", cudaGetErrorString(err), file, line);
		if (abort) {
			throw(err);
		}
	}
}
#else

// Runtime error-checker

#define cudaAssert(expression) { cudaErrCheck((expression), __FILE__, __LINE__); }

inline void cudaErrCheck(cudaError_t err, const char* file, unsigned line, bool abort = true) {
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaAssert: error %s at %s, Line %d \n", cudaGetErrorString(err), file, line);
		if (abort) {
			throw(err);
		}
	}
}

#endif // NDEBUG

#endif // !CUDA_ASSERT_H
