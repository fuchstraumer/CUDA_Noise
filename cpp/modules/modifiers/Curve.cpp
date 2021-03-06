#include "Curve.h"
#include "../cuda/modifiers/curve.cuh"
namespace cnoise {

	namespace modifiers {

		Curve::Curve(int width, int height) : Module(width, height) {}

		Curve::Curve(int width, int height, const std::vector<ControlPoint>& init_points) : Module(width, height), controlPoints(init_points) {}

		void Curve::AddControlPoint(float input_val, float output_val) {
			controlPoints.push_back(ControlPoint(input_val, output_val));
		}

		size_t Curve::GetSourceModuleCount() const {
			return 1;
		}

		std::vector<ControlPoint> Curve::GetControlPoints() const {
			return controlPoints;
		}

		void Curve::SetControlPoints(const std::vector<ControlPoint>& pts) {
			controlPoints.clear();
			controlPoints = pts;
		}

		void Curve::ClearControlPoints() {
			controlPoints.clear();
			controlPoints.shrink_to_fit();
		}

		void Curve::Generate() {
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

			CurveLauncher(Output, sourceModules.front()->Output, dims.first, dims.second, controlPoints);
			Generated = true;
		}


		Curve3D::Curve3D(Module3D* source, int width, int height) : Module3D(source, width, height) {}

		Curve3D::Curve3D(Module3D* source, int width, int height, const std::vector<ControlPoint>& init_points) : Module3D(source, width, height), controlPoints(init_points) {}

		void Curve3D::AddControlPoint(float input_val, float output_val) {
			controlPoints.push_back(ControlPoint(input_val, output_val));
		}

		void Curve3D::AddControlPoint(const ControlPoint & pt){
			controlPoints.push_back(pt);
		}

		size_t Curve3D::GetSourceModuleCount() const {
			return 1;
		}

		std::vector<ControlPoint> Curve3D::GetControlPoints() const {
			return controlPoints;
		}

		void Curve3D::SetControlPoints(const std::vector<ControlPoint>& pts) {
			controlPoints.clear();
			controlPoints = pts;
		}

		void Curve3D::ClearControlPoints() {
			controlPoints.clear();
			controlPoints.shrink_to_fit();
		}

		void Curve3D::Generate() {
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

			CurveLauncher3D(Points, sourceModules.front()->Points, dimensions.x, dimensions.y, controlPoints);
			Generated = true;
		}

	}

}