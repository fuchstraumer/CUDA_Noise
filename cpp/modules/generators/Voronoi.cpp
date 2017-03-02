#include "Voronoi.h"


namespace noise {
	
	namespace module {
		Voronoi::Voronoi(int width, int height, voronoi_distance_t cell_dist_func, voronoi_return_t return_t, float freq, float displ, int seed) : Module(width, height), Displacement(displ), Frequency(freq), CellDistanceType(cell_dist_func), ReturnDataType(return_t) {}

		int Voronoi::GetSourceModuleCount() const{
			return 0;
		}

		void Voronoi::Generate()
		{
		}
	}
}