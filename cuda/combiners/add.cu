#include "Add.cuh"

__global__ void AddKernel(float* output, float* input0, float* input1, const int width, const int height) {
	// Get current pixel.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}
	// Reading from a surface still requires the byte offset, so multiply the x coordinate by the size of a float in bytes.
	// surf2Dread also writes the value at the point to a pre-existing variable, so declare soemthing like "prev" and pass
	// it as a reference (&prev) to the surf2Dread function.
	float prev0 = input0[(j * width) + i];
	float prev1 = input1[(j * width) + i];
	// Add values and store in output.
	output[(j * width) + i] = prev0 + prev1;
}

__global__ void AddKernel3D(cnoise::Point* left, const cnoise::Point* right, const int width, const int height) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= width || j >= height) {
		return;
	}

	float prev0, prev1;
	prev0 = left[i + (j * width)].Value;
	prev1 = right[i + (j * width)].Value;

	left[i + (j * width)].Value = prev0 + prev1;
}

void AddLauncher(float* output, float* input0, float* input1, const int width, const int height){
#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	dim3 block(32, 32, 1);
	dim3 grid((width - 1) / block.x + 1, (height - 1) / block.y + 1, 1);
	AddKernel<<<grid, block>>>(output, input0, input1, width, height);
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
	printf("Add Kernel execution time in ms: %f\n", elapsed);
#endif // CUDA_KERNEL_TIMING
}

void AddLauncher3D(cnoise::Point* left, const cnoise::Point* right, const int width, const int height){

#ifdef CUDA_KERNEL_TIMING
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif // CUDA_KERNEL_TIMING

	// Setup dimensions of kernel launch using occupancy calculator.
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	AddKernel3D<<<grid, block >>>(left, right, width, height);
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
