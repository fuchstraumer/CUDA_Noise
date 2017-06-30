#include "Base.h"
#include "cuda_assert.h"
#include "../image/Image.h"
#include "../cuda/utility/normalize.cuh"

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

	void Module::ConnectModule(Module * module) {
		sourceModules.push_back(std::shared_ptr<Module>(module));
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
		cudaAssert(cudaDeviceSynchronize());
		float* norm;
		cudaAssert(cudaMallocManaged(&norm, dims.first * dims.second * sizeof(float)));
		NormalizeLauncher(norm, Output, dims.first, dims.second);
		cudaAssert(cudaDeviceSynchronize());
		return std::vector<float>(norm, norm + (dims.first * dims.second));
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

	void Module::SaveRaw32(const char* filename) {
		std::vector<float> raw = GetData();
		img::ImageWriter out(dims.first, dims.second);
		out.SetRawData(raw);
		out.WriteRaw32(filename);
	}

	void Module::SaveToTER(const char * name) {
		std::vector<float> rawData = GetData();
		img::ImageWriter out(dims.first, dims.second);
		out.SetRawData(rawData);
		out.WriteTER(name);
	}

	Module3D::Module3D(Module3D* source, int width, int height) : Generated(false), dimensions(make_int2(width, height)) {
		if (source == nullptr) {
			cudaError_t err = cudaSuccess;
			err = cudaDeviceSynchronize();
			cudaAssert(err);
			err = cudaMallocManaged(&Points, width * height * sizeof(Point));
			cudaAssert(err);
			err = cudaDeviceSynchronize();
			cudaAssert(err);
		}
		else {
			// Points is shared between modules: cuda synchronization
			// call ensures each module can write to it once they launch.
			Points = source->Points;
			sourceModules.push_back(std::shared_ptr<Module3D>(source));
		}
	}

	Module3D::Module3D(Module3D * left, Module3D * right, int width, int height) : Generated(false), dimensions(make_int2(width, height)) {
		sourceModules.push_back(std::shared_ptr<Module3D>(left));
		sourceModules.push_back(std::shared_ptr<Module3D>(right));
		Points = left->Points;
	}


	Module3D::Module3D(Module3D * left, Module3D * right, Module3D * control, const int & width, const int & height) : Generated(false), dimensions(make_int2(width, height)) {
		Points = left->Points;
		sourceModules.insert(sourceModules.end(), { std::shared_ptr<Module3D>(left), std::shared_ptr<Module3D>(right), std::shared_ptr<Module3D>(control) });
	}

	Module3D::~Module3D(){
		if (sourceModules.empty()) {
			// Synchronize device before freeing device memory
			cudaError_t err = cudaSuccess;
			err = cudaDeviceSynchronize();
			cudaAssert(err);
			// Free managed memory
			err = cudaFree(Points);
			cudaAssert(err);
			// Resync
			err = cudaDeviceSynchronize();
			cudaAssert(err);
		}
	}

	void Module3D::ConnectModule(Module3D * other){
		if (sourceModules.size() <= GetSourceModuleCount()) {
			sourceModules.push_back(std::shared_ptr<Module3D>(other));
		}
		else {
			return;
		}
	}

	Module3D* Module3D::GetModule(size_t idx) const {
		return nullptr;
	}

	void Module3D::PropagateDataset(const Point * pts){
		// Copy pts to this module
		cudaError_t err = cudaSuccess;
		err = cudaMemcpy(Points, pts, dimensions.x * dimensions.y * sizeof(Point), cudaMemcpyDefault);
		cudaAssert(err);
		// Recurse through "sourceModules" to propagate points to children.
		for (int i = 0; i < GetSourceModuleCount(); ++i) {
			if (sourceModules[i] == nullptr) {
				throw;
			}
			sourceModules[i]->PropagateDataset(pts);
		}
	}

	void Module3D::PropagateDataset(){
		for (int i = 0; i < GetSourceModuleCount(); ++i) {
			if (sourceModules[i] == nullptr) {
				throw;
			}
			cudaError_t err = cudaSuccess;
			err = cudaMemcpy(this->Points, sourceModules[i]->Points, dimensions.x * dimensions.y * sizeof(Point), cudaMemcpyDefault);
			cudaAssert(err);
			sourceModules[i]->PropagateDataset();
		}
	}

	void Module3D::freeChildren(){
		for (int i = 0; i < GetSourceModuleCount(); ++i) {
			sourceModules[i]->freeChildren();
			cudaError_t err = cudaSuccess;
			// Sync device before trying to erase, as usual.
			err = cudaDeviceSynchronize();
			cudaAssert(err);
			err = cudaFree(sourceModules[i]->Points);
			cudaAssert(err);
			sourceModules[i].reset();
		}

	}

	size_t Module3D::GetNumPts() const {
		return dimensions.x * dimensions.y;
	}

	int2 Module3D::GetDimensions() const{
		return dimensions;
	}

	std::vector<float> Module3D::GetPointValues() const{
		std::vector<float> result;
		result.resize(dimensions.x * dimensions.y);
		for (size_t j = 0; j < dimensions.y; ++j) {
			for (size_t i = 0; i < dimensions.x; ++i) {
				result[i + (j * dimensions.y)] = Points[i + (j * dimensions.y)].Value;
			}
		}
		return result;
	}

	std::vector<Point> Module3D::GetPoints() const {
		return std::vector<Point>(Points, Points + (dimensions.x * dimensions.y));
	}

	void Module3D::SaveToPNG(const char * filename) const{
		const auto data = GetPointValues();
		img::ImageWriter out(dimensions.x, dimensions.y);
		out.SetRawData(data);
		out.WritePNG(filename);
	}
}

