#include "curve.cuh"

__device__ int clamp(int val, int lower_bound, int upper_bound) {
	if (val < lower_bound) {
		return lower_bound;
	}
	else if (val > upper_bound) {
		return upper_bound;
	}
	else {
		return val;
	}
}

__device__ float cubicInterp(float n0, float n1, float n2, float n3, float a){
	float p = (n3 - n2) - (n0 - n1);
	float q = (n0 - n1) - p;
	float r = n2 - n0;
	float s = n1;
	return p * a * a * a + q * a * a + r * a + s;
}

__global__ void CurveKernel(cudaSurfaceObject_t output, cudaSurfaceObject_t input, const int width, const int height, ControlPoint* control_points, size_t num_pts) {
	// Get current pos and return if out of bounds.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= width) {
		return;
	}

	// Get previous value.
	float prev;
	surf2Dread(&prev, input, i * sizeof(float), j);

	// Get appropriate control point.
	size_t idx;
	for (idx = 0; idx < num_pts; ++idx) {
		if (prev < control_points[idx].InputVal) {
			// Found appropriate index.
			break;
		}
	}

	// Get next four nearest control points so we can interpolate.
	size_t i0, i1, i2, i3;
	i0 = clamp(idx - 2, 0, num_pts - 1);
	i1 = clamp(idx - 1, 0, num_pts - 1);
	i2 = clamp(idx, 0, num_pts - 1);
	i3 = clamp(idx + 1, 0, num_pts - 1);

	// If we don't have enough control points, just write control point value to output
	if (i1 = i2) {
		float val = control_points[i1].OutputVal;
		surf2Dwrite(val, output, i * sizeof(float), j);
		return;
	}

	// Compute alpha value used for the cubic interpolation
	float input0 = control_points[i1].InputVal;
	float input1 = control_points[i2].InputVal;
	float alpha = (prev - input0) / (input1 - input0);

	// Perform the interpolation.
	float result = cubicInterp(control_points[i0].OutputVal, control_points[i1].OutputVal, control_points[i2].OutputVal, control_points[i3].OutputVal, alpha);
	
	// Write result.
	surf2Dwrite(result, output, i * sizeof(float), j);
}

void CurveLauncher(cudaSurfaceObject_t output, cudaSurfaceObject_t input, const int width, const int height, std::vector<ControlPoint>& control_points) {

#ifdef CUDA_TIMING_TESTS
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_TIMING_TESTS

	// Setup structs on GPU
	ControlPoint *device_point_array;
	cudaMalloc(&device_point_array, control_points.size() * sizeof(ControlPoint));

	// Copy structs to GPU
	cudaMemcpy(device_point_array, &control_points[0], control_points.size() * sizeof(ControlPoint), cudaMemcpyHostToDevice);

	// Setup dimensions of kernel launch. 
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	
	// Launch kernel.
	CurveKernel<<<numBlocks, threadsPerBlock>>>(output, input, width, height, device_point_array, control_points.size());

	// Check for succesfull kernel launch
	cudaAssert(cudaGetLastError());
	// Synchronize device
	cudaAssert(cudaDeviceSynchronize());

	// Free control points array
	cudaFree(device_point_array);

#ifdef CUDA_TIMING_TESTS
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_TIMING_TESTS


}