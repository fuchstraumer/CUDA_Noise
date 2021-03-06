#include "min.cuh"

__global__ void MinKernel(float* output, const float* in0, const float* in1, const int width, const int height) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	float out_val = in0[(j * width) + i] < in1[(j * width) + i] ? in0[(j * width) + i] : in1[(j * width) + i];
	output[(j * width) + i] = out_val;
}

__global__ void MinKernel3D(cnoise::Point* output, const cnoise::Point* in0, const cnoise::Point* in1, const int width, const int height) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	output[i + (j * width)].Value = in0[i + (j * width)].Value < in1[i + (j * width)].Value ? in0[i + (j * width)].Value : in1[i + (j * width)].Value;
}

void MinLauncher(float* output, const float* in0, const float* in1, const int width, const int height) {

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	dim3 block(32, 32, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	MinKernel<<<grid, block>>>(output, in0, in1, width, height);
	// Check for succesfull kernel launch
	cudaError_t err = cudaGetLastError();
	cudaAssert(err);
	// Synchronize device
	err = cudaDeviceSynchronize();
	cudaAssert(err);

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Min Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

}

void MinLauncher3D(cnoise::Point * output, const cnoise::Point * in0, const cnoise::Point * in1, const int width, const int height){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	dim3 block(32, 32, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	MinKernel3D<<<grid, block>>>(left, right, width, height);
	// Check for succesfull kernel launch
	cudaError_t err = cudaGetLastError();
	cudaAssert(err);
	// Synchronize device
	err = cudaDeviceSynchronize();
	cudaAssert(err);

#ifdef CUDA_KERNEL_TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

}
