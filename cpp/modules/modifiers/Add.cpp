#include "Add.h"
#include "../cuda/modifiers/Add.cuh"
namespace noise {

	namespace module {

		Add::Add(int width, int height, float add_value, Module* source) : Module(width, height), addValue(add_value) {
			sourceModules.push_back(source);
		}

		void Add::Generate(){
			if (!sourceModules.front()->Generated) {
				sourceModules.front()->Generate();
			}
			if (sourceModules.front() == nullptr) {
				std::cerr << "Did you forget to attach a source module to your add module?" << std::endl;
				throw("No source module for Add module, abort");
			}
			AddLauncher(output, sourceModules[0]->output, dims.first, dims.second, addValue);
		}

		int Add::GetSourceModuleCount() const{
			return 0;
		}

		void Add::SetAddValue(float val) {
			addValue = val;
		}

		float Add::GetAddValue() const {
			return addValue;
		}

	}
}

