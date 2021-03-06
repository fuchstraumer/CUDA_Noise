#include "max.cuh"

__global__ void MaxKernel(float *output, const float *in0, const float *in1, const int width, const int height) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}

	float out_val = in0[(j * width) + i] > in1[(j * width) + i] ? in0[(j * width) + i] : in1[(j * width) + i];
	output[(j * width) + i] = out_val;
}

__global__ void MaxKernel3D(cnoise::Point* left, const cnoise::Point* right, const int width, const int height) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	float prev0, prev1;
	prev0 = left[i + (j * width)].Value;
	prev1 = right[i + (j * width)].Value;
	left[i + (j * width)].Value = (prev0 > prev1) ? prev0 : prev1;
}

void MaxLauncher(float *output, const float *in0, const float *in1, const int width, const int height) {

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	int blockSize, minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MaxKernel, 0, 0);
	dim3 block(blockSize, blockSize, 1);
	dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
	MaxKernel<<<grid, block>>>(output, in0, in1, width, height);
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
	printf("Max Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING

}

void MaxLauncher3D(cnoise::Point* left, const cnoise::Point* right, const int width, const int height){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, width / block.y, 1);
	MaxKernel3D<<<grid, block >>>(left, right, width, height);
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
