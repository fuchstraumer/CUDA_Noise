#include "Base.h"
#include "cuda_assert.h"
#include "../image/Image.h"


namespace noise {

	namespace module {

		Module::Module(int width, int height) : dims(width, height) {
			Generated = false;
			// Setup cudaSurfaceObject_t and cudaTextureObject_t objects
			// based on given dimensions.

			// They will have dimensions of width x height, and use one 
			// 32-bit channel of data.

			// Channel format description.
			cudaChannelFormatDesc cfDescr = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

			// Resource description
			struct cudaResourceDesc soDesc;
			memset(&soDesc, 0, sizeof(soDesc));
			
			// Allocate for arrays.
			// cudaMallocArray(&texArray, &cfDescr, width, height);
			// Flags only needed for surface object, in order for surface object to work.
			cudaError_t err = cudaSuccess;
			err = cudaMallocArray(&surfArray, &cfDescr, width, height, cudaArraySurfaceLoadStore);
			cudaAssert(err);

			// Now set resource description attributes.
			soDesc.resType = cudaResourceTypeArray;
			soDesc.res.array.array = surfArray;

			// Setup surface object that will be used as output data for this module to write to
			err = cudaCreateSurfaceObject(&output, &soDesc);
			cudaAssert(err);

		}

		Module::~Module() {
			cudaError_t err = cudaSuccess;
			// Free arrays
			err = cudaFreeArray(surfArray);
			cudaAssert(err);
			// Destroy objects
			err = cudaDestroySurfaceObject(output);
			cudaAssert(err);
		}

		void Module::ConnectModule(Module &other) {
			// Copy preceding source modules from "other"
			sourceModules = other.sourceModules;
			// Add other to the source modules.
			sourceModules.push_back(std::shared_ptr<Module>(&other));
		}

		cudaSurfaceObject_t Module::GetData() const{
			return output;
		}

		std::shared_ptr<Module> Module::GetModule(size_t idx) const {
			// .at(idx) has bounds checking in debug modes, iirc.
			return sourceModules.at(idx);
		}

		std::vector<float> Module::GetGPUData() const{
			// Setup result vector, allocate spacing so that memcpy succeeds.
			std::vector<float> result;
			result.resize(dims.first * dims.second);

			cudaError_t err = cudaSuccess;
			// Memcpy from device back to host
			err = cudaMemcpyFromArray(result.data(), surfArray, 0, 0, sizeof(float) * result.size(), cudaMemcpyDeviceToHost);
			cudaAssert(err);
			// Return result data.
			return result;
		}

		std::vector<float> Module::GetGPUDataNormalized() const{
			std::vector<float> tmpBuffer;
			std::vector<float> raw = GetGPUData();
			tmpBuffer.resize(raw.size());
			auto min_max = std::minmax_element(raw.begin(), raw.end());
			float max = *min_max.first;
			float min = *min_max.second;
			// std::cerr << "max: " << max << " min: " << min << std::endl;
			auto scaleRaw = [max, min](float val)->float {
				float result;
				result = (val - min) / (max - min);
				return result;
			};
			std::transform(raw.begin(), raw.end(), tmpBuffer.begin(), scaleRaw);
			return tmpBuffer;
		}

		void Module::SaveToPNG(const char * name){
			std::vector<float> rawData = GetGPUData();
			ImageWriter out(dims.first, dims.second);
			out.SetRawData(std::move(rawData));
			out.ConvertRawData();
			out.WritePNG(name);
		}



	}
}

