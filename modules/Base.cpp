#include "stdafx.h"
#include "Base.h"

namespace noise {
	namespace module {

		Module::Module(int width, int height) : dims(width, height) {
			// Setup cudaSurfaceObject_t and cudaTextureObject_t objects
			// based on given dimensions.

			// They will have dimensions of width x height, and use one 
			// 32-bit channel of data.

			// Channel format description.
			cudaChannelFormatDesc cfDescr = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

			// Resource description
			struct cudaResourceDesc soDesc;
			memset(&soDesc, 0, sizeof(soDesc));
			struct cudaResourceDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			
			// Allocate for arrays.
			cudaMallocArray(&texArray, &cfDescr, width, height);
			// Flags only needed for surface object, in order for surface object to work.
			cudaMallocArray(&surfArray, &cfDescr, width, height, cudaArraySurfaceLoadStore);

			// Now set resource description attributes.
			soDesc.resType = cudaResourceTypeArray;
			texDesc.resType = cudaResourceTypeArray;
			soDesc.res.array.array = surfArray;
			texDesc.res.array.array = texArray;

			// Specify texture data and params (bad name, but we already used "texDesc")
			struct cudaTextureDesc texTDescr;
			memset(&texTDescr, 0, sizeof(texTDescr));
			texTDescr.readMode = cudaReadModeElementType;

			// Setup texture object that will be used as input data from the previous module.
			input = 0;
			cudaCreateTextureObject(&input, &texDesc, &texTDescr, nullptr);

			// Setup surface object that will be used as output data for this module to write to
			output = 0;
			cudaCreateSurfaceObject(&output, &soDesc);

		}

		Module::~Module() {
			// Destroy objects
			cudaDestroySurfaceObject(output);
			cudaDestroyTextureObject(input);
			// Free arrays
			cudaFreeArray(surfArray);
			cudaFreeArray(texArray);
		}

		void Module::ConnectModule(Module &other) {
			// Copy preceding source modules from "other"
			sourceModules = other.sourceModules;
			// Add other to the source modules.
			sourceModules.push_back(std::shared_ptr<Module>(&other));
		}

		std::shared_ptr<Module> Module::GetModule(size_t idx) const {
			// .at(idx) has bounds checking in debug modes, iirc.
			return sourceModules.at(idx);
		}


	}
}

