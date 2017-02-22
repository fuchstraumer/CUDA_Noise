#include "Curve.h"

namespace noise {
	namespace module {
		Curve::Curve(int width, int height) : Module(width, height) {

		}

		Curve::Curve(int width, int height, const std::vector<ControlPoint>& init_points) : Module(width, height) {

		}

		Curve::~Curve(){

		}

		void Curve::PrepareData(){
			if (Ready && !update) {
				// Don't want to do this stuff if we're already good to go.
				return;
			}
			else {
				// CUDA data not ready, need to get thigns onto the GPU and prepare for kernel launch.

				// Create channel format description: two channels, each 32-bit floats.
				cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

				// Allocate array
				cudaError_t err = cudaSuccess;
				err = cudaMallocArray(&cpArray, &cfDesc, controlPoints.size(), 2);
				cudaAssert(err);

				// Copy data to array.
				err = cudaMemcpyToArray(cpArray, 0, 0, &controlPoints[0], controlPoints.size() * 2 * sizeof(float), cudaMemcpyHostToDevice);
				cudaAssert(err);

				// Create resource description, used to tie array and texture together
				struct cudaResourceDesc rDesc;
				memset(&rDesc, 0, sizeof(rDesc));

				// Set rDesc attributs
				rDesc.resType = cudaResourceTypeArray;
				rDesc.res.array.array = cpArray;

				// Create texture description
				struct cudaTextureDesc tDesc;
				memset(&tDesc, 0, sizeof(tDesc));

				// Specify read type, filtering, border/wrapping

				// Don't allow edge wrapping or looping, clamp to edges so out-of-range values
				// become edge values.
				tDesc.addressMode[0] = cudaAddressModeClamp;
				tDesc.addressMode[1] = cudaAddressModeClamp;
				tDesc.addressMode[2] = cudaAddressModeClamp;

				// No filtering, this is important to set. Otherwise our values we want to be exact will be linearly interpolated.
				tDesc.filterMode = cudaFilterModePoint;

				// Don't make the int data in this texture floating-point. Only counts for the CUDA-exclusive elements of the code.
				// Data is still 32 bits per pixel/element, and if we copy it back to the CPU there's nothing stopping us from treating
				// it like floating-point data.
				tDesc.readMode = cudaReadModeElementType;

				// Don't normalize coordinates.
				tDesc.normalizedCoords = false;

				// Last step, create texture object
				cpTex = 0;
				err = cudaCreateTextureObject(&cpTex, &rDesc, &tDesc, nullptr);
				cudaAssert(err);

			}
		}

		void Curve::AddControlPoint(float input_val, float output_val){
			controlPoints.push_back(ControlPoint(input_val, output_val));
			if (!update) {
				// Flag that we need to update.
				update = true;
			}
		}

		std::vector<ControlPoint> Curve::GetControlPoints() const{
			return controlPoints;
		}

		void Curve::ClearControlPoints(){
			controlPoints.clear();
			controlPoints.shrink_to_fit();
		}

		void noise::module::Curve::ConnectModule(Module & other){
			std::shared_ptr<Module> ptr(&other);
			sourceModules.push_back(ptr);
		}

		void Curve::Generate(){
			if (!Ready || update) {
				if (controlPoints.empty()) {
					std::cerr << "Data unprepared in a Curve module, and control point vector empty! Exiting." << std::endl;
					throw("NOISE::MODULES::MODIFIER::CURVE:L103: Control points vector empty.");
				}
				else {
					PrepareData();
				}
			}

		}
	}
}