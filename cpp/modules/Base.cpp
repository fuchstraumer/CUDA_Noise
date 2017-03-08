#include "Base.h"
#include "cuda_assert.h"
#include "../image/Image.h"


namespace cnoise {
	
	Module::Module(int width, int height) : dims(width, height) {
			Generated = false;
			// Allocate using managed memory, so that CPU/GPU can share a single pointer.
			// Be sure to call cudaDeviceSynchronize() before accessing Output.
			cudaError_t err = cudaSuccess;
			err = cudaMallocManaged(&Output, sizeof(float) * width * height);
			cudaAssert(err);

			// Synchronize device to make sure we can access the Output pointer freely and safely.
			err = cudaDeviceSynchronize();
			cudaAssert(err);
		}

		Module::~Module() {
			cudaError_t err = cudaSuccess;
			// Synchronize device to make sure its not doing anything with the elements we wish to destroy
			err = cudaDeviceSynchronize();
			cudaAssert(err);
			// Free managed memory.
			err = cudaFree(Output);
			cudaAssert(err);
		}

		void Module::ConnectModule(Module* other) {
			if (sourceModules.size() < GetSourceModuleCount()) {
				sourceModules.push_back(other);
			}
		}

		std::vector<float> Module::GetData() const{
			// Make sure to sync device before trying to get data.
			cudaAssert(cudaDeviceSynchronize());
			std::vector<float> result(Output, Output + (dims.first * dims.second));
			return result;
		}

		Module& Module::GetModule(size_t idx) const {
			// .at(idx) has bounds checking in debug modes, iirc.
			return *sourceModules.at(idx);
		}

		std::vector<float> Module::GetDataNormalized(float upper_bound, float lower_bound) const{
			return std::vector<float>();
		}

		void Module::SaveToPNG(const char * name){
			std::vector<float> rawData = GetData();
			img::ImageWriter out(dims.first, dims.second);
			out.SetRawData(rawData);
			out.WritePNG(name);
		}

		void Module::SaveToPNG_16(const char * filename) {
			std::vector<float> raw = GetData();
			img::ImageWriter out(dims.first, dims.second);
			out.SetRawData(raw);
			out.WritePNG_16(filename);
		}

		void Module::SaveToTER(const char * name) {
			std::vector<float> rawData = GetData();
			img::ImageWriter out(dims.first, dims.second);
			out.SetRawData(rawData);
			out.WriteTER(name);
		}

		Module3D::Module3D(int width, int height, int depth) : Dimensions(make_int3(width, height, depth)) {
			Generated = false;
			// Allocate using managed memory, so that CPU/GPU can share a single pointer.
			// Be sure to call cudaDeviceSynchronize() before accessing Output.
			cudaError_t err = cudaSuccess;
			err = cudaMallocManaged(&Output, sizeof(float) * width * height * depth);
			cudaAssert(err);

			// Synchronize device to make sure we can access the Output pointer freely and safely.
			err = cudaDeviceSynchronize();
			cudaAssert(err);
		}

		Module3D::~Module3D(){
			cudaAssert(cudaDeviceSynchronize());
			cudaAssert(cudaFree(Output));
		}

		void Module3D::ConnectModule(Module3D * other){
			if (sourceModules.size() <= GetSourceModuleCount()) {
				sourceModules.push_back(other);
			}
			else {
				return;
			}
		}

		std::vector<float> Module3D::GetData() const{
			// Make sure to sync to device before accessing managed memory.
			cudaAssert(cudaDeviceSynchronize());
			return std::vector<float>(Output, Output + (Dimensions.x * Dimensions.y * Dimensions.z));
		}

		Module3D* Module3D::GetModule(size_t idx) const{
			return sourceModules.at(idx);
		}

		std::vector<float> Module3D::GetDataNormalized(float upper_bound, float lower_bound) const{
			return std::vector<float>();
		}

}

