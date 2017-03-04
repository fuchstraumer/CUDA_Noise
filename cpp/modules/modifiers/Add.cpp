#include "Add.h"
#include "../cuda/modifiers/add.cuh"
namespace noise {

	namespace module {

		Add::Add(int width, int height, float add_value, Module* source) : Module(width, height), addValue(add_value) {
			sourceModules.push_back(source);
		}

		void Add::Generate(){
			// Check source modules container.
			if (sourceModules.empty() || sourceModules.front() == nullptr) {
				std::cerr << "Did you forget to attach a source module(s) to your add module?" << std::endl;
				throw("No source module(s) for Add module, abort");
			}

			// Make sure all modules in source modules container are generated.
			for (const auto& module : sourceModules) {
				if (!module->Generated) {
					module->Generate();
				}
			}
			
			AddLauncher(output, sourceModules[0]->output, dims.first, dims.second, addValue);
		}

		size_t Add::GetSourceModuleCount() const{
			return sourceModules.size();
		}

		void Add::SetAddValue(float val) {
			addValue = val;
		}

		float Add::GetAddValue() const {
			return addValue;
		}

	}
}

