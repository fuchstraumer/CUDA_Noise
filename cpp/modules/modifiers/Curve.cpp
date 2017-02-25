#include "Curve.h"
#include "../cuda/modifiers/curve.cuh"
namespace noise {
	namespace module {

		Curve::Curve(int width, int height) : Module(width, height) {}

		Curve::Curve(int width, int height, const std::vector<ControlPoint>& init_points) : Module(width, height), controlPoints(init_points) {}

		void Curve::AddControlPoint(float input_val, float output_val){
			controlPoints.push_back(ControlPoint(input_val, output_val));
		}

		std::vector<ControlPoint> Curve::GetControlPoints() const{
			return controlPoints;
		}

		void Curve::ClearControlPoints(){
			controlPoints.clear();
			controlPoints.shrink_to_fit();
		}

		void Curve::Generate(){
			if (controlPoints.empty()) {
				std::cerr << "Data unprepared in a Curve module, and control point vector empty! Exiting." << std::endl;
				throw("NOISE::MODULES::MODIFIER::CURVE:L103: Control points vector empty.");
			}
			if (sourceModules.front() == nullptr) {
				std::cerr << "Did you forget to attach a module for curving to this curve module?" << std::endl;
				throw("NOISE::MODULES::MODIFIER::CURVE:L32: No source module connected.");
			}
			if (!sourceModules.front()->Generated) {
				sourceModules.front()->Generate();
			}
			CurveLauncher(output, sourceModules.front()->output, dims.first, dims.second, controlPoints);
			Generated = true;
		}
	}
}