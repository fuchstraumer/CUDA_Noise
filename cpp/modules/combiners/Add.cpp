#include "Add.h"
#include "../cuda/combiners/add.cuh"
namespace cnoise {

	namespace combiners {

		Add::Add(int width, int height, float add_value, Module* source) : Module(width, height) {
			sourceModules.push_back(std::shared_ptr<Module>(source));
		}

		void Add::Generate(){
			// Make sure all modules in source modules container are generated. (and that there are no null module connections)
			for (const auto& module : sourceModules) {
				if (module == nullptr) {
					std::cerr << "Did you forget to attach a source module(s) to your add module?" << std::endl;
					throw("No source module(s) for Add module, abort");
				}
				if (!module->Generated) {
					module->Generate();
				}
			}
			
			AddLauncher(Output, sourceModules[0]->Output, sourceModules[1]->Output, dims.first, dims.second);
			Generated = true;
		}

		size_t Add::GetSourceModuleCount() const{
			return 2;
		}

		Add3D::Add3D(Module3D* left, Module3D* right, const int& width, const int& height) : Module3D(left, right, width, height) {}

		void Add3D::Generate() {
			for (const auto& module : sourceModules) {
				if (!module) {
					std::cerr << "Module not supplied to Add3D module: a source module was nullptr!";
					throw std::runtime_error("Invalid/null module given to Add3D Module");
				}
				if (!module->Generated) {
					module->Generate();
				}
			}

			AddLauncher3D(sourceModules[0]->Points, sourceModules[1]->Points, dimensions.x, dimensions.y);
			Generated = true;
		}

		size_t Add3D::GetSourceModuleCount() const {
			return 2;
		}

	}
}

