#include "Sphere.h"

namespace cnoise {
	
	namespace models {

		Sphere::Sphere(size_t width, size_t height, float east_lon_bound, float west_lon_bound, float south_latt_bound, float north_latt_bound) : east(east_lon_bound), west(west_lon_bound),
		north(north_latt_bound), south(south_latt_bound), dimensions(make_int2(width, height)) {
			// Allocate array of geocoords
			cudaAssert(cudaDeviceSynchronize());
			cudaError_t err = cudaSuccess;
			err = cudaMallocManaged(&points, width * height * sizeof(GeoCoord));
			cudaAssert(err);
			// Synchronize device to make sure malloc completes on host and device.
			cudaAssert(cudaDeviceSynchronize());
		}

		Sphere::Sphere(Module3D * _source, float east_lon_bound, float west_lon_bound, float south_latt_bound, float north_latt_bound) : east(east_lon_bound), west(west_lon_bound),
			north(north_latt_bound), south(south_latt_bound), dimensions(_source->GetDimensions()), source(_source) {
			// Allocate array of geocoords
			cudaAssert(cudaDeviceSynchronize());
			cudaError_t err = cudaSuccess;
			const size_t num_pts = source->GetNumPts();
			err = cudaMallocManaged(&points, num_pts * sizeof(GeoCoord));
			cudaAssert(err);
			// Synchronize device to make sure malloc completes on host and device.
			cudaAssert(cudaDeviceSynchronize());
		}

		Sphere::~Sphere(){
			// Synchronize device to make sure coords isn't in use.
			cudaAssert(cudaDeviceSynchronize());
			cudaError_t err = cudaSuccess;
			err = cudaFree(points);
			cudaAssert(err);
		}

		void Sphere::SaveToPNG(const char* filename) const {
			source->SaveToPNG(filename);
		}

		void Sphere::Build(){
			// Invalid parameter check.
			if (north <= south || east <= west) {
				throw;
			}

			// Step size in geographic terms.
			float d_long, d_latt;
			d_long = (east - west) / static_cast<float>(dimensions.x);
			d_latt = (north - south) / static_cast<float>(dimensions.y);

			// Current geographic sampling location.
			float longitude, lattitude;
			longitude = west;
			lattitude = south;

			// Setup geocoords.
			for (int y = 0; y < dimensions.y; ++y) {
				longitude = west;
				for (int x = 0; x < dimensions.x; ++x) {
					points[x + (dimensions.y * y)] = GeoCoord(lattitude, longitude);
					longitude += d_long;
				}
				lattitude += d_latt;
			}
			// Before generating, propagate the points we setup to all connected modules.
			source->PropagateDataset(points);

			// Call "source", and hopefully our points propagated...
			source->Generate();
		}

		void Sphere::SetSourceModule(Module3D * src){
			source = src;
		}

		void Sphere::SetEasternBound(float _east){
			east = _east;
		}

		void Sphere::SetWesternBound(float _west){
			west = _west;
		}

		void Sphere::SetSouthernBound(float _south){
			south = _south;
		}

		void Sphere::SetNorthernBound(float _north){
			north = _north;
		}

		float Sphere::GetEasternBound() const{
			return east;
		}

		float Sphere::GetWesternBound() const{
			return west;
		}

		float Sphere::GetSouthernBound() const{
			return south;
		}

		float Sphere::GetNorthernBound() const{
			return north;
		}

	}
}
